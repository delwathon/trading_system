"""
Database Manager for the Enhanced Bybit Trading System.
Replaces CSV export functionality with comprehensive MySQL database storage.
"""

import logging
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, and_

from database.models import (
    DatabaseManager, ScanSession, TradingSignal, TradingOpportunity,
    MarketSummary, PerformanceMetric, SystemLog
)


def json_serializer(obj):
    """Custom JSON serializer for numpy and other non-serializable types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif hasattr(obj, 'isoformat'):  # datetime objects
        return obj.isoformat()
    return obj


def safe_json_data(data):
    """Safely convert data to JSON-serializable format"""
    if data is None:
        return None
    if isinstance(data, (dict, list)):
        try:
            # First convert numpy types, then ensure it's JSON serializable
            converted = json.loads(json.dumps(data, default=json_serializer))
            return converted
        except (TypeError, ValueError) as e:
            logging.warning(f"JSON serialization warning: {e}")
            return {}
    return data


class EnhancedDatabaseManager:
    """Enhanced database manager that replaces CSV functionality with MySQL storage"""
    
    def __init__(self, database_manager: DatabaseManager):
        self.db_manager = database_manager
        self.logger = logging.getLogger(__name__)
        
    def create_scan_session(self, scan_id: str, **kwargs) -> int:
        """Create a new scan session and return its ID"""
        try:
            session = self.db_manager.get_session()
            
            scan_session = ScanSession(
                scan_id=scan_id,
                timestamp=kwargs.get('timestamp', datetime.utcnow()),
                execution_time_seconds=kwargs.get('execution_time_seconds'),
                symbols_analyzed=kwargs.get('symbols_analyzed'),
                signals_generated=kwargs.get('signals_generated'),
                success_rate=kwargs.get('success_rate'),
                charts_generated=kwargs.get('charts_generated'),
                parallel_processing=kwargs.get('parallel_processing', False),
                threads_used=kwargs.get('threads_used'),
                mtf_enabled=kwargs.get('mtf_enabled', False),
                primary_timeframe=kwargs.get('primary_timeframe', '15m'),
                confirmation_timeframes=kwargs.get('confirmation_timeframes', ['1h', '4h']),
                mtf_weight_multiplier=kwargs.get('mtf_weight_multiplier')
            )
            
            session.add(scan_session)
            session.commit()
            session_id = scan_session.id
            session.close()
            
            primary_tf = kwargs.get('primary_timeframe', '15m')
            confirmation_tfs = ', '.join(kwargs.get('confirmation_timeframes', ['1h', '4h']))
            self.logger.debug(f"✅ Created scan session: {scan_id} (ID: {session_id}) - Primary: {primary_tf}, Confirmation: {confirmation_tfs}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Error creating scan session: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return None
    
    def save_signals(self, signals: List[Dict], scan_session_id: int) -> bool:
        """Save trading signals to MySQL database"""
        try:
            if not signals:
                self.logger.warning("No signals to save")
                return False
            
            session = self.db_manager.get_session()
            rows_added = 0
            
            for signal in signals:
                # Get MTF analysis data
                mtf_analysis = signal.get('mtf_analysis', {})
                analysis = signal.get('analysis', {})
                risk_assessment = analysis.get('risk_assessment', {})
                
                # Create TradingSignal record
                signal_record = TradingSignal(
                    scan_session_id=scan_session_id,
                    symbol=signal.get('symbol', ''),
                    side=signal.get('side', '').lower(),
                    order_type=signal.get('order_type', '').lower(),
                    entry_price=float(signal.get('entry_price', 0)),
                    current_price=float(signal.get('current_price', 0)),
                    stop_loss=float(signal.get('stop_loss', 0)),
                    take_profit_1=float(signal.get('take_profit_1', 0)),
                    take_profit_2=float(signal.get('take_profit_2')) if signal.get('take_profit_2') else None,
                    confidence=float(signal.get('confidence', 0)),
                    original_confidence=float(signal.get('original_confidence', signal.get('confidence', 0))),
                    mtf_boost=float(signal.get('confidence', 0) - signal.get('original_confidence', signal.get('confidence', 0))),
                    mtf_status=signal.get('mtf_status', 'UNKNOWN'),
                    mtf_confirmed_timeframes=safe_json_data(mtf_analysis.get('confirmed_timeframes', [])),
                    mtf_conflicting_timeframes=safe_json_data(mtf_analysis.get('conflicting_timeframes', [])),
                    mtf_confirmation_count=int(len(mtf_analysis.get('confirmed_timeframes', []))),
                    mtf_confirmation_strength=float(mtf_analysis.get('confirmation_strength', 0)),
                    risk_reward_ratio=float(signal.get('risk_reward_ratio', 0)),
                    risk_level=risk_assessment.get('risk_level', 'Unknown'),
                    total_risk_score=float(risk_assessment.get('total_risk_score', 0)),
                    volume_24h=float(signal.get('volume_24h', 0)),
                    price_change_24h=float(signal.get('price_change_24h', 0)),
                    signal_type=signal.get('signal_type', ''),
                    chart_file=signal.get('chart_file', ''),
                    priority_boost=int(signal.get('priority_boost', 0)),
                    technical_score=float(analysis.get('technical_summary', {}).get('trend', {}).get('score', 0)),
                    volume_score=float(analysis.get('technical_summary', {}).get('volume', {}).get('ratio', 1)),
                    fibonacci_score=0.0,  # Placeholder
                    confluence_zones_count=int(len(analysis.get('confluence_zones', []))),
                    technical_summary=safe_json_data(analysis.get('technical_summary', {})),
                    volume_profile_data=safe_json_data(analysis.get('volume_profile', {})),
                    fibonacci_data=safe_json_data(analysis.get('fibonacci_data', {})),
                    confluence_zones=safe_json_data(analysis.get('confluence_zones', [])),
                    entry_methods=safe_json_data(signal.get('entry_methods', {}))
                )
                
                session.add(signal_record)
                rows_added += 1
            
            session.commit()
            session.close()
            
            self.logger.debug(f"✅ Saved {rows_added} signals to MySQL database")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving signals to MySQL: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False
    
    def save_opportunities(self, opportunities: List[Dict], scan_session_id: int) -> bool:
        """Save trading opportunities to MySQL database"""
        try:
            if not opportunities:
                self.logger.warning("No opportunities to save")
                return False
            
            session = self.db_manager.get_session()
            rows_added = 0
            
            for opp in opportunities:
                opportunity_record = TradingOpportunity(
                    scan_session_id=scan_session_id,
                    rank=int(opp.get('rank', 0)),
                    symbol=opp.get('symbol', ''),
                    side=opp.get('side', '').lower(),
                    order_type=opp.get('order_type', '').lower(),
                    confidence=float(opp.get('confidence', 0)),
                    original_confidence=float(opp.get('original_confidence', opp.get('confidence', 0))),
                    mtf_boost=float(opp.get('mtf_boost', 0)),
                    entry_price=float(opp.get('entry_price', 0)),
                    current_price=float(opp.get('current_price', 0)),
                    stop_loss=float(opp.get('stop_loss', 0)),
                    take_profit_1=float(opp.get('take_profit_1', opp.get('take_profit', 0))),
                    take_profit_2=float(opp.get('take_profit_2', 0)) if opp.get('take_profit_2') else None,
                    risk_reward_ratio=float(opp.get('risk_reward_ratio', 0)),
                    volume_24h=float(opp.get('volume_24h', 0)),
                    total_score=float(opp.get('total_score', 0)),
                    mtf_status=opp.get('mtf_status', 'UNKNOWN'),
                    mtf_confirmed=safe_json_data(opp.get('mtf_confirmed', [])),
                    mtf_conflicting=safe_json_data(opp.get('mtf_conflicting', [])),
                    mtf_confirmation_count=int(opp.get('mtf_confirmation_count', 0)),
                    mtf_total_timeframes=int(opp.get('mtf_total_timeframes', 0)),
                    mtf_confirmation_strength=float(opp.get('mtf_confirmation_strength', 0)),
                    priority_boost=int(opp.get('priority_boost', 0)),
                    risk_level=opp.get('risk_level', 'Unknown'),
                    chart_file=opp.get('chart_file', ''),
                    signal_type=opp.get('signal_type', ''),
                    distance_from_current=float(opp.get('distance_from_current', 0)),
                    volume_score=0.0,  # Placeholder
                    technical_strength=0.0  # Placeholder
                )
                
                session.add(opportunity_record)
                rows_added += 1
            
            session.commit()
            session.close()
            
            self.logger.debug(f"✅ Saved {rows_added} opportunities to MySQL database")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving opportunities to MySQL: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False
    
    def save_market_summary(self, results: Dict, scan_session_id: int) -> bool:
        """Save market summary to MySQL database"""
        try:
            session = self.db_manager.get_session()
            
            scan_info = results.get('scan_info', {})
            market_summary = results.get('market_summary', {})
            performance = results.get('system_performance', {})
            
            # Count MTF signal types
            signals = results.get('signals', [])
            mtf_strong = len([s for s in signals if s.get('mtf_status') == 'STRONG'])
            mtf_partial = len([s for s in signals if s.get('mtf_status') == 'PARTIAL'])
            mtf_none = len([s for s in signals if s.get('mtf_status') == 'NONE'])
            
            # Get top movers
            top_movers = market_summary.get('top_movers', {})
            biggest_gainer = top_movers.get('biggest_gainer', {})
            biggest_loser = top_movers.get('biggest_loser', {})
            highest_volume = top_movers.get('highest_volume', {})
            
            market_summary_record = MarketSummary(
                scan_session_id=scan_session_id,
                total_market_volume=float(market_summary.get('total_market_volume', 0)),
                average_volume=float(market_summary.get('average_volume', 0)),
                market_sentiment_bullish_pct=float(market_summary.get('market_sentiment', {}).get('bullish_percentage', 0)),
                buy_signals=int(market_summary.get('signal_distribution', {}).get('buy_signals', 0)),
                sell_signals=int(market_summary.get('signal_distribution', {}).get('sell_signals', 0)),
                market_orders=int(market_summary.get('signal_distribution', {}).get('market_orders', 0)),
                limit_orders=int(market_summary.get('signal_distribution', {}).get('limit_orders', 0)),
                signals_per_minute=float(performance.get('signals_per_minute', 0)),
                avg_confidence=float(performance.get('avg_confidence', 0)),
                avg_original_confidence=float(performance.get('avg_original_confidence', 0)),
                mtf_boost_avg=float(performance.get('mtf_boost_avg', 0)),
                speedup_factor=float(performance.get('speedup_factor', 1.0)),
                mtf_strong_signals=int(mtf_strong),
                mtf_partial_signals=int(mtf_partial),
                mtf_none_signals=int(mtf_none),
                top_gainer_symbol=biggest_gainer.get('symbol', ''),
                top_gainer_change=float(biggest_gainer.get('price_change_24h', 0)),
                top_loser_symbol=biggest_loser.get('symbol', ''),
                top_loser_change=float(biggest_loser.get('price_change_24h', 0)),
                highest_volume_symbol=highest_volume.get('symbol', ''),
                market_sentiment_data=safe_json_data(market_summary.get('market_sentiment', {})),
                volume_distribution=safe_json_data(performance.get('order_type_distribution', {})),
                mtf_analysis_summary=safe_json_data(market_summary.get('mtf_analysis', {}))
            )
            
            session.add(market_summary_record)
            session.commit()
            session.close()
            
            self.logger.debug("✅ Saved market summary to MySQL database")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving market summary to MySQL: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False
    
    def save_performance_metrics(self, metrics: List[Dict], scan_session_id: int) -> bool:
        """Save performance metrics to MySQL database"""
        try:
            if not metrics:
                return True
            
            session = self.db_manager.get_session()
            rows_added = 0
            
            for metric in metrics:
                metric_record = PerformanceMetric(
                    scan_session_id=scan_session_id,
                    metric_type=metric.get('type', ''),
                    metric_name=metric.get('name', ''),
                    value=metric.get('value', 0),
                    unit=metric.get('unit', ''),
                    timeframe=metric.get('timeframe', ''),
                    symbol=metric.get('symbol', ''),
                    additional_data=metric.get('additional_data', {})
                )
                
                session.add(metric_record)
                rows_added += 1
            
            session.commit()
            session.close()
            
            self.logger.debug(f"✅ Saved {rows_added} performance metrics to MySQL database")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving performance metrics to MySQL: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False
    
    def save_all_results(self, results: Dict) -> Dict[str, Any]:
        """Save all results to MySQL database (replaces CSV export)"""
        try:
            scan_id = self.generate_scan_id()
            self.logger.debug(f"📊 Saving results to MySQL database for scan ID: {scan_id}")
            
            # Create scan session
            scan_info = results.get('scan_info', {})
            scan_session_id = self.create_scan_session(
                scan_id=scan_id,
                timestamp=scan_info.get('timestamp', datetime.utcnow()),
                execution_time_seconds=scan_info.get('execution_time_seconds'),
                symbols_analyzed=scan_info.get('symbols_analyzed'),
                signals_generated=scan_info.get('signals_generated'),
                success_rate=scan_info.get('success_rate'),
                charts_generated=scan_info.get('charts_generated'),
                parallel_processing=scan_info.get('parallel_processing'),
                threads_used=scan_info.get('threads_used'),
                mtf_enabled=scan_info.get('mtf_enabled'),
                primary_timeframe=scan_info.get('primary_timeframe', '15m'),
                confirmation_timeframes=scan_info.get('confirmation_timeframes', ['1h', '4h']),
                mtf_weight_multiplier=scan_info.get('mtf_weight_multiplier')
            )
            
            if not scan_session_id:
                return {'error': 'Failed to create scan session'}
            
            saved_data = {'scan_id': scan_id, 'scan_session_id': scan_session_id}
            
            # Get timeframe info for logging
            primary_tf = scan_info.get('primary_timeframe', '15m')
            confirmation_tfs = ', '.join(scan_info.get('confirmation_timeframes', ['1h', '4h']))
            
            # Save signals
            signals = results.get('signals', [])
            if signals and self.save_signals(signals, scan_session_id):
                saved_data['signals'] = len(signals)
            
            # Save opportunities
            opportunities = results.get('top_opportunities', [])
            if opportunities and self.save_opportunities(opportunities, scan_session_id):
                saved_data['opportunities'] = len(opportunities)
            
            # Save market summary
            if self.save_market_summary(results, scan_session_id):
                saved_data['market_summary'] = True
            
            # Save performance metrics if available
            performance_metrics = results.get('performance_metrics', [])
            if performance_metrics and self.save_performance_metrics(performance_metrics, scan_session_id):
                saved_data['performance_metrics'] = len(performance_metrics)
            
            self.logger.debug(f"📄 MySQL Database Save Complete (Scan ID: {scan_id}):")
            self.logger.debug(f"   Primary Timeframe: {primary_tf}")
            self.logger.debug(f"   Confirmation Timeframes: {confirmation_tfs}")
            self.logger.debug(f"   Signals: {saved_data.get('signals', 0)}")
            self.logger.debug(f"   Opportunities: {saved_data.get('opportunities', 0)}")
            self.logger.debug(f"   Market Summary: {'✅' if saved_data.get('market_summary') else '❌'}")
            self.logger.debug(f"   Performance Metrics: {saved_data.get('performance_metrics', 0)}")
            
            return saved_data
            
        except Exception as e:
            self.logger.error(f"MySQL database save error: {e}")
            return {'error': str(e)}
    
    def generate_scan_id(self) -> str:
        """Generate unique scan ID with timestamp"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def get_recent_scans(self, limit: int = 10) -> List[Dict]:
        """Get recent scan sessions from MySQL"""
        try:
            session = self.db_manager.get_session()
            
            scans = session.query(ScanSession).order_by(
                desc(ScanSession.timestamp)
            ).limit(limit).all()
            
            result = []
            for scan in scans:
                result.append({
                    'scan_id': scan.scan_id,
                    'timestamp': scan.timestamp,
                    'execution_time_seconds': scan.execution_time_seconds,
                    'symbols_analyzed': scan.symbols_analyzed,
                    'signals_generated': scan.signals_generated,
                    'success_rate': scan.success_rate,
                    'charts_generated': scan.charts_generated,
                    'mtf_enabled': scan.mtf_enabled,
                    'primary_timeframe': scan.primary_timeframe,
                    'confirmation_timeframes': scan.confirmation_timeframes
                })
            
            session.close()
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting recent scans from MySQL: {e}")
            return []
    
    def get_signals(self, scan_id: str = None, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Get trading signals with optional filters from MySQL"""
        try:
            session = self.db_manager.get_session()
            
            query = session.query(TradingSignal).join(ScanSession)
            
            if scan_id:
                query = query.filter(ScanSession.scan_id == scan_id)
            if symbol:
                query = query.filter(TradingSignal.symbol == symbol)
            
            signals = query.order_by(desc(TradingSignal.timestamp)).limit(limit).all()
            
            result = []
            for signal in signals:
                result.append({
                    'scan_id': signal.scan_session.scan_id,
                    'timestamp': signal.timestamp,
                    'symbol': signal.symbol,
                    'side': signal.side,
                    'order_type': signal.order_type,
                    'entry_price': signal.entry_price,
                    'current_price': signal.current_price,
                    'stop_loss': signal.stop_loss,
                    'take_profit_1': signal.take_profit_1,
                    'take_profit_2': signal.take_profit_2,
                    'confidence': signal.confidence,
                    'original_confidence': signal.original_confidence,
                    'mtf_boost': signal.mtf_boost,
                    'mtf_status': signal.mtf_status,
                    'mtf_confirmed_timeframes': signal.mtf_confirmed_timeframes,
                    'risk_reward_ratio': signal.risk_reward_ratio,
                    'volume_24h': signal.volume_24h,
                    'chart_file': signal.chart_file,
                    'primary_timeframe': signal.scan_session.primary_timeframe,
                    'confirmation_timeframes': signal.scan_session.confirmation_timeframes
                })
            
            session.close()
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting signals from MySQL: {e}")
            return []
    
    def get_database_stats(self) -> Dict:
        """Get comprehensive MySQL database statistics"""
        try:
            session = self.db_manager.get_session()
            
            # Count records in each table
            stats = {
                'scan_sessions': session.query(ScanSession).count(),
                'trading_signals': session.query(TradingSignal).count(),
                'trading_opportunities': session.query(TradingOpportunity).count(),
                'market_summaries': session.query(MarketSummary).count(),
                'performance_metrics': session.query(PerformanceMetric).count(),
                'system_logs': session.query(SystemLog).count()
            }
            
            # Get latest scan info
            latest_scan = session.query(ScanSession).order_by(
                desc(ScanSession.timestamp)
            ).first()
            
            if latest_scan:
                stats['latest_scan'] = {
                    'scan_id': latest_scan.scan_id,
                    'timestamp': latest_scan.timestamp,
                    'signals_generated': latest_scan.signals_generated,
                    'execution_time': latest_scan.execution_time_seconds,
                    'primary_timeframe': latest_scan.primary_timeframe,
                    'confirmation_timeframes': latest_scan.confirmation_timeframes
                }
            
            # Get signal statistics
            signal_stats = session.query(
                TradingSignal.side,
                func.count(TradingSignal.id).label('count'),
                func.avg(TradingSignal.confidence).label('avg_confidence')
            ).group_by(TradingSignal.side).all()
            
            stats['signal_distribution'] = {
                stat.side: {
                    'count': stat.count,
                    'avg_confidence': float(stat.avg_confidence) if stat.avg_confidence else 0
                }
                for stat in signal_stats
            }
            
            # MTF analysis statistics
            mtf_stats = session.query(
                TradingSignal.mtf_status,
                func.count(TradingSignal.id).label('count')
            ).group_by(TradingSignal.mtf_status).all()
            
            stats['mtf_distribution'] = {
                stat.mtf_status: stat.count
                for stat in mtf_stats
            }
            
            session.close()
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting MySQL database stats: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old data from MySQL database tables"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            cleanup_stats = {}
            
            self.logger.debug(f"🧹 Cleaning up MySQL data older than {days_to_keep} days (before {cutoff_date})")
            
            session = self.db_manager.get_session()
            
            # Delete old scan sessions and related data (cascade will handle related records)
            old_sessions = session.query(ScanSession).filter(
                ScanSession.timestamp < cutoff_date
            ).all()
            
            sessions_to_delete = len(old_sessions)
            
            if sessions_to_delete > 0:
                # Count related records before deletion
                signals_to_delete = session.query(TradingSignal).filter(
                    TradingSignal.scan_session_id.in_([s.id for s in old_sessions])
                ).count()
                
                opportunities_to_delete = session.query(TradingOpportunity).filter(
                    TradingOpportunity.scan_session_id.in_([s.id for s in old_sessions])
                ).count()
                
                # Delete old data
                session.query(TradingSignal).filter(
                    TradingSignal.scan_session_id.in_([s.id for s in old_sessions])
                ).delete(synchronize_session=False)
                
                session.query(TradingOpportunity).filter(
                    TradingOpportunity.scan_session_id.in_([s.id for s in old_sessions])
                ).delete(synchronize_session=False)
                
                session.query(MarketSummary).filter(
                    MarketSummary.scan_session_id.in_([s.id for s in old_sessions])
                ).delete(synchronize_session=False)
                
                session.query(ScanSession).filter(
                    ScanSession.timestamp < cutoff_date
                ).delete(synchronize_session=False)
                
                session.commit()
                
                cleanup_stats = {
                    'scan_sessions': sessions_to_delete,
                    'trading_signals': signals_to_delete,
                    'trading_opportunities': opportunities_to_delete
                }
                
                self.logger.debug(f"   Scan Sessions: removed {sessions_to_delete}")
                self.logger.debug(f"   Trading Signals: removed {signals_to_delete}")
                self.logger.debug(f"   Trading Opportunities: removed {opportunities_to_delete}")
            
            # Clean up old system logs
            old_logs_count = session.query(SystemLog).filter(
                SystemLog.timestamp < cutoff_date
            ).count()
            
            if old_logs_count > 0:
                session.query(SystemLog).filter(
                    SystemLog.timestamp < cutoff_date
                ).delete(synchronize_session=False)
                
                cleanup_stats['system_logs'] = old_logs_count
                self.logger.debug(f"   System Logs: removed {old_logs_count}")
            
            session.commit()
            session.close()
            
            total_removed = sum(cleanup_stats.values())
            self.logger.info(f"🧹 MySQL cleanup complete: removed {total_removed} total records")
            
            return cleanup_stats
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old MySQL data: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return {}
    
    def log_system_event(self, level: str, message: str, scan_session_id: int = None, 
                        symbol: str = None, function_name: str = None, 
                        error_type: str = None, stack_trace: str = None):
        """Log system events to MySQL database"""
        try:
            session = self.db_manager.get_session()
            
            log_record = SystemLog(
                level=level.upper(),
                logger_name='TradingSystem',
                message=message,
                scan_session_id=scan_session_id,
                symbol=symbol,
                function_name=function_name,
                error_type=error_type,
                stack_trace=stack_trace
            )
            
            session.add(log_record)
            session.commit()
            session.close()
            
        except Exception as e:
            # Don't log errors from logging to avoid recursion
            print(f"Error saving log to MySQL database: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
    
    def get_mtf_performance_report(self, days: int = 7) -> Dict:
        """Get Multi-Timeframe performance report for the last N days"""
        try:
            session = self.db_manager.get_session()
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Get recent signals with MTF data
            recent_signals = session.query(TradingSignal).join(ScanSession).filter(
                ScanSession.timestamp >= cutoff_date
            ).all()
            
            report = {
                'period_days': days,
                'total_signals': len(recent_signals),
                'timeframe_config': {
                    'primary': recent_signals[0].scan_session.primary_timeframe if recent_signals else '15m',
                    'confirmation': recent_signals[0].scan_session.confirmation_timeframes if recent_signals else ['1h', '4h']
                },
                'mtf_breakdown': {
                    'STRONG': 0,
                    'PARTIAL': 0,
                    'NONE': 0,
                    'DISABLED': 0
                },
                'confidence_analysis': {
                    'avg_original_confidence': 0,
                    'avg_final_confidence': 0,
                    'avg_mtf_boost': 0
                },
                'best_performers': [],
                'symbols_analyzed': len(set([s.symbol for s in recent_signals]))
            }
            
            if recent_signals:
                # MTF breakdown
                for signal in recent_signals:
                    mtf_status = signal.mtf_status or 'UNKNOWN'
                    if mtf_status in report['mtf_breakdown']:
                        report['mtf_breakdown'][mtf_status] += 1
                    else:
                        report['mtf_breakdown']['UNKNOWN'] = report['mtf_breakdown'].get('UNKNOWN', 0) + 1
                
                # Confidence analysis
                original_confidences = [s.original_confidence for s in recent_signals if s.original_confidence]
                final_confidences = [s.confidence for s in recent_signals if s.confidence]
                mtf_boosts = [s.mtf_boost for s in recent_signals if s.mtf_boost]
                
                if original_confidences:
                    report['confidence_analysis']['avg_original_confidence'] = sum(original_confidences) / len(original_confidences)
                if final_confidences:
                    report['confidence_analysis']['avg_final_confidence'] = sum(final_confidences) / len(final_confidences)
                if mtf_boosts:
                    report['confidence_analysis']['avg_mtf_boost'] = sum(mtf_boosts) / len(mtf_boosts)
                
                # Best performers (highest confidence + MTF boost)
                strong_signals = [s for s in recent_signals if s.mtf_status == 'STRONG']
                strong_signals.sort(key=lambda x: x.confidence, reverse=True)
                
                for signal in strong_signals[:5]:  # Top 5
                    report['best_performers'].append({
                        'symbol': signal.symbol,
                        'side': signal.side,
                        'confidence': signal.confidence,
                        'mtf_boost': signal.mtf_boost,
                        'confirmed_timeframes': signal.mtf_confirmed_timeframes,
                        'timestamp': signal.timestamp
                    })
            
            session.close()
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating MTF performance report: {e}")
            return {}
    
    def export_signals_to_csv(self, scan_id: str = None, file_path: str = None) -> str:
        """Export signals from MySQL to CSV for backup/analysis"""
        try:
            import csv
            
            if file_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = f"signal_export_{timestamp}.csv"
            
            signals = self.get_signals(scan_id=scan_id, limit=1000)
            
            if not signals:
                self.logger.warning("No signals found for export")
                return ""
            
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                if signals:
                    writer = csv.DictWriter(f, fieldnames=signals[0].keys())
                    writer.writeheader()
                    writer.writerows(signals)
            
            self.logger.debug(f"✅ Exported {len(signals)} signals to {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error exporting signals to CSV: {e}")
            return ""