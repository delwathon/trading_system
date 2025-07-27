"""
Database Manager for the Enhanced Bybit Trading System.
Replaces CSV export functionality with comprehensive MySQL database storage.
UPDATED: Added auto-trading position and session management.
"""

import logging
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, and_

from utils.logging import get_logger
from database.models import (
    DatabaseManager, ScanSession, TradingSignal, TradingOpportunity,
    MarketSummary, PerformanceMetric, SystemLog, TradingPosition, AutoTradingSession
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
        self.logger = get_logger(__name__)
    
    # ===== AUTO-TRADING SESSION MANAGEMENT =====
    
    def create_auto_trading_session(self, session_id: str, config_snapshot: Dict) -> int:
        """Create new auto-trading session"""
        try:
            session = self.db_manager.get_session()
            
            auto_session = AutoTradingSession(
                session_id=session_id,
                config_snapshot=safe_json_data(config_snapshot),
                status='active'
            )
            
            session.add(auto_session)
            session.commit()
            session_id_db = auto_session.id
            session.close()
            
            self.logger.debug(f"✅ Created auto-trading session: {session_id} (ID: {session_id_db})")
            return session_id_db
            
        except Exception as e:
            self.logger.error(f"Error creating auto-trading session: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return None
    
    def update_auto_trading_session_stats(self, session_id: str, **kwargs) -> bool:
        """Update auto-trading session statistics"""
        try:
            session = self.db_manager.get_session()
            
            auto_session = session.query(AutoTradingSession).filter(
                AutoTradingSession.session_id == session_id,
                AutoTradingSession.status == 'active'
            ).first()
            
            if auto_session:
                for key, value in kwargs.items():
                    if hasattr(auto_session, key):
                        setattr(auto_session, key, value)
                
                session.commit()
                session.close()
                return True
            
            session.close()
            return False
            
        except Exception as e:
            self.logger.error(f"Error updating auto-trading session: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False
    
    def end_auto_trading_session(self, session_id: str, final_stats: Dict = None) -> bool:
        """End auto-trading session with final statistics"""
        try:
            session = self.db_manager.get_session()
            
            auto_session = session.query(AutoTradingSession).filter(
                AutoTradingSession.session_id == session_id,
                AutoTradingSession.status == 'active'
            ).first()
            
            if auto_session:
                auto_session.ended_at = datetime.utcnow()
                auto_session.status = 'stopped'
                
                if final_stats:
                    for key, value in final_stats.items():
                        if hasattr(auto_session, key):
                            setattr(auto_session, key, value)
                
                session.commit()
                session.close()
                
                self.logger.info(f"✅ Ended auto-trading session: {session_id}")
                return True
            
            session.close()
            return False
            
        except Exception as e:
            self.logger.error(f"Error ending auto-trading session: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False
    
    # ===== POSITION MANAGEMENT =====
    
    def save_trading_position(self, position_data: Dict, scan_session_id: int = None) -> str:
        """Save trading position to database"""
        try:
            session = self.db_manager.get_session()
            
            position = TradingPosition(
                scan_session_id=scan_session_id,
                position_id=position_data['position_id'],
                symbol=position_data['symbol'],
                side=position_data['side'],
                entry_price=float(position_data['entry_price']),
                position_size=float(position_data['position_size']),
                leverage=str(position_data['leverage']),
                risk_amount=float(position_data['risk_amount']),
                entry_order_id=position_data.get('entry_order_id'),
                stop_loss_order_id=position_data.get('stop_loss_order_id'),
                take_profit_order_id=position_data.get('take_profit_order_id'),
                stop_loss_price=float(position_data.get('stop_loss', 0)),
                take_profit_price=float(position_data.get('take_profit', 0)),
                auto_close_profit_target=float(position_data.get('auto_close_profit_target', 10.0)),
                signal_confidence=float(position_data.get('signal_confidence', 0)),
                mtf_status=position_data.get('mtf_status', ''),
                status='open'
            )
            
            session.add(position)
            session.commit()
            
            position_id = position_data['position_id']
            session.close()
            
            self.logger.debug(f"✅ Saved trading position: {position_id}")
            return position_id
            
        except Exception as e:
            self.logger.error(f"Error saving trading position: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return ""
    
    def update_position_status(self, position_id: str, status: str, **kwargs) -> bool:
        """Update position status and other fields"""
        try:
            session = self.db_manager.get_session()
            
            position = session.query(TradingPosition).filter(
                TradingPosition.position_id == position_id
            ).first()
            
            if position:
                position.status = status
                position.last_updated = datetime.utcnow()
                
                if status == 'closed':
                    position.closed_at = datetime.utcnow()
                
                # Update additional fields
                for key, value in kwargs.items():
                    if hasattr(position, key):
                        setattr(position, key, value)
                
                session.commit()
                session.close()
                
                self.logger.debug(f"✅ Updated position {position_id} status to {status}")
                return True
            
            session.close()
            return False
            
        except Exception as e:
            self.logger.error(f"Error updating position status: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False
    
    def get_active_positions(self) -> List[Dict]:
        """Get all active trading positions"""
        try:
            session = self.db_manager.get_session()
            
            positions = session.query(TradingPosition).filter(
                TradingPosition.status == 'open'
            ).all()
            
            result = []
            for pos in positions:
                result.append({
                    'position_id': pos.position_id,
                    'symbol': pos.symbol,
                    'side': pos.side,
                    'entry_price': pos.entry_price,
                    'position_size': pos.position_size,
                    'leverage': pos.leverage,
                    'risk_amount': pos.risk_amount,
                    'stop_loss_price': pos.stop_loss_price,
                    'take_profit_price': pos.take_profit_price,
                    'auto_close_profit_target': pos.auto_close_profit_target,
                    'opened_at': pos.opened_at,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_pct': pos.unrealized_pnl_pct,
                    'leveraged_pnl_pct': pos.leveraged_pnl_pct
                })
            
            session.close()
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting active positions: {e}")
            return []
    
    def get_position_history(self, limit: int = 100, symbol: str = None) -> List[Dict]:
        """Get position trading history"""
        try:
            session = self.db_manager.get_session()
            
            query = session.query(TradingPosition)
            
            if symbol:
                query = query.filter(TradingPosition.symbol == symbol)
            
            positions = query.order_by(
                desc(TradingPosition.opened_at)
            ).limit(limit).all()
            
            result = []
            for pos in positions:
                result.append({
                    'position_id': pos.position_id,
                    'symbol': pos.symbol,
                    'side': pos.side,
                    'entry_price': pos.entry_price,
                    'position_size': pos.position_size,
                    'leverage': pos.leverage,
                    'risk_amount': pos.risk_amount,
                    'status': pos.status,
                    'opened_at': pos.opened_at,
                    'closed_at': pos.closed_at,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_pct': pos.unrealized_pnl_pct,
                    'leveraged_pnl_pct': pos.leveraged_pnl_pct,
                    'auto_closed': pos.auto_closed,
                    'close_reason': pos.close_reason,
                    'signal_confidence': pos.signal_confidence,
                    'mtf_status': pos.mtf_status
                })
            
            session.close()
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting position history: {e}")
            return []
    
    # ===== SCAN SESSION MANAGEMENT (UPDATED) =====
        
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
                primary_timeframe=kwargs.get('primary_timeframe', '30m'),
                confirmation_timeframes=kwargs.get('confirmation_timeframes', ['1h', '4h', '6h']),
                mtf_weight_multiplier=kwargs.get('mtf_weight_multiplier'),
                # NEW: Auto-trading fields
                auto_trading_session_id=kwargs.get('auto_trading_session_id'),
                trades_executed_count=kwargs.get('trades_executed_count', 0)
            )
            
            session.add(scan_session)
            session.commit()
            session_id = scan_session.id
            session.close()
            
            primary_tf = kwargs.get('primary_timeframe', '30m')
            confirmation_tfs = ', '.join(kwargs.get('confirmation_timeframes', ['1h', '4h', '6h']))
            self.logger.debug(f"✅ Created scan session: {scan_id} (ID: {session_id}) - Primary: {primary_tf}, Confirmation: {confirmation_tfs}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Error creating scan session: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return None
    
    def save_signals(self, signals: List[Dict], scan_session_id: int) -> bool:
        """Save trading signals to MySQL database with auto-trading fields"""
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
                    # NEW: Auto-trading execution tracking
                    selected_for_execution=signal.get('selected_for_execution', False),
                    execution_attempted=signal.get('execution_attempted', False),
                    execution_successful=signal.get('execution_successful', False),
                    execution_error=signal.get('execution_error'),
                    position_id=signal.get('position_id'),
                    # Analysis data
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
        """Save trading opportunities to MySQL database with auto-trading fields"""
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
                    technical_strength=0.0,  # Placeholder
                    # NEW: Auto-trading execution tracking
                    selected_for_execution=opp.get('selected_for_execution', False),
                    execution_attempted=opp.get('execution_attempted', False),
                    execution_successful=opp.get('execution_successful', False)
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
    
    def save_all_results(self, results: Dict, auto_trading_session_id: int = None) -> Dict[str, Any]:
        """Save all results to MySQL database with auto-trading session link"""
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
                primary_timeframe=scan_info.get('primary_timeframe', '30m'),
                confirmation_timeframes=scan_info.get('confirmation_timeframes', ['1h', '4h', '6h']),
                mtf_weight_multiplier=scan_info.get('mtf_weight_multiplier'),
                # NEW: Link to auto-trading session
                auto_trading_session_id=auto_trading_session_id,
                trades_executed_count=scan_info.get('trades_executed_count', 0)
            )
            
            if not scan_session_id:
                return {'error': 'Failed to create scan session'}
            
            saved_data = {'scan_id': scan_id, 'scan_session_id': scan_session_id}
            
            # Get timeframe info for logging
            primary_tf = scan_info.get('primary_timeframe', '30m')
            confirmation_tfs = ', '.join(scan_info.get('confirmation_timeframes', ['1h', '4h', '6h']))
            
            # FIXED: Only save TOP OPPORTUNITIES (not all signals)
            # Get the top opportunities that are actually displayed on console
            top_opportunities = results.get('top_opportunities', [])
        
            if top_opportunities:
                # Save top opportunities as signals (these are the ones displayed)
                if self.save_signals(top_opportunities, scan_session_id):
                    saved_data['signals'] = len(top_opportunities)
                
                # Save the same top opportunities in opportunities table
                if self.save_opportunities(top_opportunities, scan_session_id):
                    saved_data['opportunities'] = len(top_opportunities)
                
                self.logger.debug(f"✅ DATABASE STORAGE: Only TOP {len(top_opportunities)} opportunities saved")
                self.logger.debug(f"   (These match exactly what's displayed in console)")
            else:
                self.logger.info("ℹ️  No top opportunities to save to database")
                saved_data['signals'] = 0
                saved_data['opportunities'] = 0
            
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
            self.logger.debug(f"   TOP Opportunities Saved: {saved_data.get('signals', 0)}")
            self.logger.debug(f"   Market Summary: {'✅' if saved_data.get('market_summary') else '❌'}")
            self.logger.debug(f"   Performance Metrics: {saved_data.get('performance_metrics', 0)}")
            if auto_trading_session_id:
                self.logger.debug(f"   Auto-Trading Session: {auto_trading_session_id}")
            
            return saved_data
            
        except Exception as e:
            self.logger.error(f"MySQL database save error: {e}")
            return {'error': str(e)}
    
    def generate_scan_id(self) -> str:
        """Generate unique scan ID with timestamp"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ===== AUTO-TRADING ANALYTICS =====
    
    def get_auto_trading_performance(self, session_id: str = None, days: int = 7) -> Dict:
        """Get auto-trading performance report"""
        try:
            session = self.db_manager.get_session()
            
            # Base query for positions
            query = session.query(TradingPosition)
            
            if session_id:
                # Get specific session performance
                auto_session = session.query(AutoTradingSession).filter(
                    AutoTradingSession.session_id == session_id
                ).first()
                
                if auto_session:
                    query = query.join(ScanSession).filter(
                        ScanSession.auto_trading_session_id == auto_session.id
                    )
            else:
                # Get performance for last N days
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                query = query.filter(TradingPosition.opened_at >= cutoff_date)
            
            positions = query.all()
            
            # Calculate performance metrics
            total_positions = len(positions)
            if total_positions == 0:
                return {'error': 'No positions found for the specified criteria'}
            
            closed_positions = [p for p in positions if p.status == 'closed']
            open_positions = [p for p in positions if p.status == 'open']
            
            profit_positions = [p for p in closed_positions if p.unrealized_pnl_pct > 0]
            loss_positions = [p for p in closed_positions if p.unrealized_pnl_pct < 0]
            
            # Calculate metrics
            win_rate = (len(profit_positions) / len(closed_positions) * 100) if closed_positions else 0
            total_pnl = sum(p.unrealized_pnl for p in closed_positions)
            avg_win = np.mean([p.unrealized_pnl_pct for p in profit_positions]) if profit_positions else 0
            avg_loss = np.mean([p.unrealized_pnl_pct for p in loss_positions]) if loss_positions else 0
            
            # Risk metrics
            total_risk_exposed = sum(p.risk_amount for p in positions)
            max_concurrent = len(open_positions)
            
            # Duration analysis
            closed_with_duration = [p for p in closed_positions if p.closed_at]
            if closed_with_duration:
                durations = [(p.closed_at - p.opened_at).total_seconds() / 60 for p in closed_with_duration]
                avg_duration_minutes = np.mean(durations)
            else:
                avg_duration_minutes = 0
            
            performance_report = {
                'period_analysis': {
                    'total_positions': total_positions,
                    'open_positions': len(open_positions),
                    'closed_positions': len(closed_positions),
                    'profit_positions': len(profit_positions),
                    'loss_positions': len(loss_positions)
                },
                'performance_metrics': {
                    'win_rate_percent': win_rate,
                    'total_pnl': total_pnl,
                    'average_win_percent': avg_win,
                    'average_loss_percent': avg_loss,
                    'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
                    'average_duration_minutes': avg_duration_minutes
                },
                'risk_metrics': {
                    'total_risk_exposed': total_risk_exposed,
                    'max_concurrent_positions': max_concurrent,
                    'risk_per_position': total_risk_exposed / total_positions if total_positions > 0 else 0
                },
                'leverage_analysis': {},
                'symbol_performance': {}
            }
            
            # Leverage analysis
            leverage_groups = {}
            for pos in positions:
                lev = pos.leverage
                if lev not in leverage_groups:
                    leverage_groups[lev] = []
                leverage_groups[lev].append(pos)
            
            for lev, pos_list in leverage_groups.items():
                closed_lev_positions = [p for p in pos_list if p.status == 'closed']
                if closed_lev_positions:
                    lev_win_rate = len([p for p in closed_lev_positions if p.unrealized_pnl_pct > 0]) / len(closed_lev_positions) * 100
                    lev_avg_pnl = np.mean([p.unrealized_pnl_pct for p in closed_lev_positions])
                    
                    performance_report['leverage_analysis'][lev] = {
                        'positions': len(pos_list),
                        'win_rate_percent': lev_win_rate,
                        'average_pnl_percent': lev_avg_pnl
                    }
            
            # Symbol performance
            symbol_groups = {}
            for pos in positions:
                sym = pos.symbol
                if sym not in symbol_groups:
                    symbol_groups[sym] = []
                symbol_groups[sym].append(pos)
            
            # Get top 10 most traded symbols
            sorted_symbols = sorted(symbol_groups.items(), key=lambda x: len(x[1]), reverse=True)[:10]
            
            for sym, pos_list in sorted_symbols:
                closed_sym_positions = [p for p in pos_list if p.status == 'closed']
                if closed_sym_positions:
                    sym_win_rate = len([p for p in closed_sym_positions if p.unrealized_pnl_pct > 0]) / len(closed_sym_positions) * 100
                    sym_avg_pnl = np.mean([p.unrealized_pnl_pct for p in closed_sym_positions])
                    
                    performance_report['symbol_performance'][sym] = {
                        'positions': len(pos_list),
                        'win_rate_percent': sym_win_rate,
                        'average_pnl_percent': sym_avg_pnl
                    }
            
            session.close()
            return performance_report
            
        except Exception as e:
            self.logger.error(f"Error getting auto-trading performance: {e}")
            return {'error': str(e)}
    
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
                    'confirmation_timeframes': scan.confirmation_timeframes,
                    'trades_executed_count': scan.trades_executed_count,
                    'auto_trading_session_id': scan.auto_trading_session_id
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
                    'confirmation_timeframes': signal.scan_session.confirmation_timeframes,
                    # Auto-trading fields
                    'selected_for_execution': signal.selected_for_execution,
                    'execution_attempted': signal.execution_attempted,
                    'execution_successful': signal.execution_successful,
                    'execution_error': signal.execution_error,
                    'position_id': signal.position_id
                })
            
            session.close()
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting signals from MySQL: {e}")
            return []
    
    def get_database_stats(self) -> Dict:
        """Get comprehensive MySQL database statistics including auto-trading data"""
        try:
            session = self.db_manager.get_session()
            
            # Count records in each table
            stats = {
                'scan_sessions': session.query(ScanSession).count(),
                'trading_signals': session.query(TradingSignal).count(),
                'trading_opportunities': session.query(TradingOpportunity).count(),
                'market_summaries': session.query(MarketSummary).count(),
                'performance_metrics': session.query(PerformanceMetric).count(),
                'system_logs': session.query(SystemLog).count(),
                # NEW: Auto-trading tables
                'trading_positions': session.query(TradingPosition).count(),
                'auto_trading_sessions': session.query(AutoTradingSession).count()
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
                    'confirmation_timeframes': latest_scan.confirmation_timeframes,
                    'trades_executed': latest_scan.trades_executed_count
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
            
            # NEW: Auto-trading statistics
            position_stats = session.query(
                TradingPosition.status,
                func.count(TradingPosition.id).label('count')
            ).group_by(TradingPosition.status).all()
            
            stats['position_distribution'] = {
                stat.status: stat.count
                for stat in position_stats
            }
            
            # Active auto-trading sessions
            active_sessions = session.query(AutoTradingSession).filter(
                AutoTradingSession.status == 'active'
            ).count()
            
            stats['active_auto_trading_sessions'] = active_sessions
            
            session.close()
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting MySQL database stats: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old data from MySQL database tables including auto-trading data"""
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
                
                positions_to_delete = session.query(TradingPosition).filter(
                    TradingPosition.scan_session_id.in_([s.id for s in old_sessions])
                ).count()
                
                # Delete old data
                session.query(TradingSignal).filter(
                    TradingSignal.scan_session_id.in_([s.id for s in old_sessions])
                ).delete(synchronize_session=False)
                
                session.query(TradingOpportunity).filter(
                    TradingOpportunity.scan_session_id.in_([s.id for s in old_sessions])
                ).delete(synchronize_session=False)
                
                session.query(TradingPosition).filter(
                    TradingPosition.scan_session_id.in_([s.id for s in old_sessions])
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
                    'trading_opportunities': opportunities_to_delete,
                    'trading_positions': positions_to_delete
                }
                
                self.logger.debug(f"   Scan Sessions: removed {sessions_to_delete}")
                self.logger.debug(f"   Trading Signals: removed {signals_to_delete}")
                self.logger.debug(f"   Trading Opportunities: removed {opportunities_to_delete}")
                self.logger.debug(f"   Trading Positions: removed {positions_to_delete}")
            
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
            
            # Clean up old completed auto-trading sessions
            old_auto_sessions = session.query(AutoTradingSession).filter(
                and_(
                    AutoTradingSession.ended_at < cutoff_date,
                    AutoTradingSession.status != 'active'
                )
            ).count()
            
            if old_auto_sessions > 0:
                session.query(AutoTradingSession).filter(
                    and_(
                        AutoTradingSession.ended_at < cutoff_date,
                        AutoTradingSession.status != 'active'
                    )
                ).delete(synchronize_session=False)
                
                cleanup_stats['auto_trading_sessions'] = old_auto_sessions
                self.logger.debug(f"   Auto-Trading Sessions: removed {old_auto_sessions}")
            
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
                logger_name='AutoTradingSystem',
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
                    'primary': recent_signals[0].scan_session.primary_timeframe if recent_signals else '30m',
                    'confirmation': recent_signals[0].scan_session.confirmation_timeframes if recent_signals else ['1h', '4h', '6h']
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
                'auto_trading_stats': {
                    'signals_executed': 0,
                    'execution_success_rate': 0,
                    'positions_opened': 0
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
                
                # Auto-trading statistics
                executed_signals = [s for s in recent_signals if s.execution_attempted]
                successful_executions = [s for s in recent_signals if s.execution_successful]
                
                report['auto_trading_stats']['signals_executed'] = len(executed_signals)
                report['auto_trading_stats']['execution_success_rate'] = (
                    len(successful_executions) / len(executed_signals) * 100 if executed_signals else 0
                )
                report['auto_trading_stats']['positions_opened'] = len(successful_executions)
                
                # Best performers (highest confidence + MTF boost)
                strong_signals = [s for s in recent_signals if s.mtf_status == 'STRONG']
                strong_signals.sort(key=lambda x: x.confidence, reverse=True)
                
                for signal in strong_signals[:5]:  # Top 5
                    report['best_performers'].append({
                        'symbol': signal.symbol,
                        'side': signal.side,
                        'confidence': signal.confidence,
                        'mtf_boost': signal.mtf_boost,
                        'mtf_confirmed_timeframes': signal.mtf_confirmed_timeframes,
                        'timestamp': signal.timestamp,
                        'executed': signal.execution_successful
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