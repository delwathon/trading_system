"""
Enhanced CSV Manager for the Enhanced Bybit Trading System.
Handles all CSV export functionality with append mode and comprehensive data tracking.
"""

import os
import csv
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from config.config import EnhancedSystemConfig


class EnhancedCSVManager:
    """Enhanced CSV manager that maintains single files and appends new data"""
    
    def __init__(self, config: EnhancedSystemConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Create CSV directory if it doesn't exist
        self.csv_dir = Path("csv_exports")
        self.csv_dir.mkdir(exist_ok=True)
        
        # Define fixed CSV filenames (no timestamps)
        self.csv_files = {
            'signals': self.csv_dir / f"{config.csv_base_filename}_signals.csv",
            'opportunities': self.csv_dir / f"{config.csv_base_filename}_opportunities.csv", 
            'market_summary': self.csv_dir / f"{config.csv_base_filename}_market_summary.csv",
            'scan_history': self.csv_dir / f"{config.csv_base_filename}_scan_history.csv",
            'performance_metrics': self.csv_dir / f"{config.csv_base_filename}_performance.csv"
        }
        
        # Initialize CSV files with headers if they don't exist
        self.initialize_csv_files()
    
    def initialize_csv_files(self):
        """Initialize CSV files with headers if they don't exist"""
        try:
            # Define headers for each CSV file
            headers = {
                'signals': [
                    'scan_id', 'timestamp', 'symbol', 'side', 'order_type', 'entry_price',
                    'current_price', 'stop_loss', 'take_profit_1', 'take_profit_2',
                    'confidence', 'original_confidence', 'mtf_boost', 'mtf_status',
                    'mtf_confirmed_timeframes', 'mtf_conflicting_timeframes',
                    'mtf_confirmation_count', 'mtf_confirmation_strength',
                    'risk_reward_ratio', 'risk_level', 'total_risk_score',
                    'volume_24h', 'price_change_24h', 'signal_type', 'chart_file',
                    'priority_boost', 'primary_timeframe', 'confirmation_timeframes',
                    'mtf_weight_multiplier', 'technical_score', 'volume_score',
                    'fibonacci_score', 'confluence_zones_count'
                ],
                'opportunities': [
                    'scan_id', 'timestamp', 'rank', 'symbol', 'side', 'order_type',
                    'confidence', 'original_confidence', 'mtf_boost', 'entry_price',
                    'current_price', 'stop_loss', 'take_profit_1', 'take_profit_2',
                    'risk_reward_ratio', 'volume_24h', 'total_score', 'mtf_status',
                    'mtf_confirmed', 'mtf_conflicting', 'mtf_confirmation_count',
                    'mtf_total_timeframes', 'mtf_confirmation_strength', 'priority_boost',
                    'risk_level', 'chart_file', 'signal_type', 'distance_from_current',
                    'volume_score', 'technical_strength'
                ],
                'market_summary': [
                    'scan_id', 'timestamp', 'execution_time_seconds', 'symbols_analyzed',
                    'signals_generated', 'success_rate', 'charts_generated',
                    'parallel_processing', 'threads_used', 'mtf_enabled',
                    'confirmation_timeframes', 'primary_timeframe', 'signals_per_minute',
                    'avg_confidence', 'avg_original_confidence', 'mtf_boost_avg',
                    'total_market_volume', 'average_volume', 'market_sentiment_bullish_pct',
                    'buy_signals', 'sell_signals', 'market_orders', 'limit_orders',
                    'speedup_factor', 'mtf_strong_signals', 'mtf_partial_signals',
                    'mtf_none_signals', 'top_gainer_symbol', 'top_gainer_change',
                    'top_loser_symbol', 'top_loser_change', 'highest_volume_symbol'
                ],
                'scan_history': [
                    'scan_id', 'timestamp', 'duration_seconds', 'symbols_scanned',
                    'signals_found', 'success_rate', 'mtf_enabled', 'primary_timeframe',
                    'confirmation_timeframes', 'charts_generated', 'csv_files_updated',
                    'system_memory_usage_mb', 'cpu_usage_percent', 'errors_encountered'
                ],
                'performance_metrics': [
                    'scan_id', 'timestamp', 'metric_type', 'metric_name', 'value',
                    'unit', 'timeframe', 'symbol', 'additional_data'
                ]
            }
            
            # Initialize each CSV file
            for file_type, file_path in self.csv_files.items():
                self._init_csv_file(file_path, headers[file_type])
            
            self.logger.info(f"✅ CSV files initialized in {self.csv_dir}")
            for file_type, file_path in self.csv_files.items():
                self.logger.info(f"   {file_type}: {file_path.name}")
            
        except Exception as e:
            self.logger.error(f"Error initializing CSV files: {e}")
    
    def _init_csv_file(self, file_path: Path, headers: List[str]):
        """Initialize a single CSV file with headers if it doesn't exist"""
        try:
            if not file_path.exists():
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                self.logger.debug(f"📄 Created new CSV file: {file_path.name}")
            else:
                self.logger.debug(f"📄 Using existing CSV file: {file_path.name}")
        except Exception as e:
            self.logger.error(f"Error initializing {file_path}: {e}")
    
    def generate_scan_id(self) -> str:
        """Generate unique scan ID with timestamp"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def append_signals(self, signals: List[Dict], scan_id: str) -> bool:
        """Append new signals to the signals CSV file"""
        try:
            if not signals:
                self.logger.warning("No signals to append")
                return False
            
            file_path = self.csv_files['signals']
            rows_added = 0
            
            with open(file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                for signal in signals:
                    row = self._build_signal_row(signal, scan_id)
                    writer.writerow(row)
                    rows_added += 1
            
            self.logger.info(f"✅ Appended {rows_added} signals to {file_path.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error appending signals: {e}")
            return False
    
    def _build_signal_row(self, signal: Dict, scan_id: str) -> List[Any]:
        """Build a signal row for CSV export"""
        # Get MTF analysis data
        mtf_analysis = signal.get('mtf_analysis', {})
        confirmed_timeframes = mtf_analysis.get('confirmed_timeframes', [])
        conflicting_timeframes = mtf_analysis.get('conflicting_timeframes', [])
        
        # Get analysis data
        analysis = signal.get('analysis', {})
        risk_assessment = analysis.get('risk_assessment', {})
        technical_summary = analysis.get('technical_summary', {})
        
        return [
            scan_id,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            signal.get('symbol', ''),
            signal.get('side', '').upper(),
            signal.get('order_type', '').upper(),
            signal.get('entry_price', 0),
            signal.get('current_price', 0),
            signal.get('stop_loss', 0),
            signal.get('take_profit_1', 0),
            signal.get('take_profit_2', 0),
            signal.get('confidence', 0),
            signal.get('original_confidence', signal.get('confidence', 0)),
            signal.get('confidence', 0) - signal.get('original_confidence', signal.get('confidence', 0)),
            signal.get('mtf_status', 'UNKNOWN'),
            ', '.join(confirmed_timeframes),
            ', '.join(conflicting_timeframes),
            len(confirmed_timeframes),
            mtf_analysis.get('confirmation_strength', 0),
            signal.get('risk_reward_ratio', 0),
            risk_assessment.get('risk_level', 'Unknown'),
            risk_assessment.get('total_risk_score', 0),
            signal.get('volume_24h', 0),
            signal.get('price_change_24h', 0),
            signal.get('signal_type', ''),
            signal.get('chart_file', ''),
            signal.get('priority_boost', 0),
            self.config.timeframe,
            ', '.join(self.config.confirmation_timeframes),
            self.config.mtf_weight_multiplier,
            technical_summary.get('trend', {}).get('score', 0),
            technical_summary.get('volume', {}).get('ratio', 1),
            0,  # fibonacci_score (placeholder)
            len(analysis.get('confluence_zones', []))
        ]
    
    def append_opportunities(self, opportunities: List[Dict], scan_id: str) -> bool:
        """Append new opportunities to the opportunities CSV file"""
        try:
            if not opportunities:
                self.logger.warning("No opportunities to append")
                return False
            
            file_path = self.csv_files['opportunities']
            rows_added = 0
            
            with open(file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                for opp in opportunities:
                    row = self._build_opportunity_row(opp, scan_id)
                    writer.writerow(row)
                    rows_added += 1
            
            self.logger.info(f"✅ Appended {rows_added} opportunities to {file_path.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error appending opportunities: {e}")
            return False
    
    def _build_opportunity_row(self, opp: Dict, scan_id: str) -> List[Any]:
        """Build an opportunity row for CSV export"""
        return [
            scan_id,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            opp.get('rank', 0),
            opp.get('symbol', ''),
            opp.get('side', ''),
            opp.get('order_type', ''),
            opp.get('confidence', 0),
            opp.get('original_confidence', opp.get('confidence', 0)),
            opp.get('mtf_boost', 0),
            opp.get('entry_price', 0),
            opp.get('current_price', 0),
            opp.get('stop_loss', 0),
            opp.get('take_profit_1', opp.get('take_profit', 0)),
            opp.get('take_profit_2', 0),
            opp.get('risk_reward_ratio', 0),
            opp.get('volume_24h', 0),
            opp.get('total_score', 0),
            opp.get('mtf_status', 'UNKNOWN'),
            ', '.join(opp.get('mtf_confirmed', [])),
            ', '.join(opp.get('mtf_conflicting', [])),
            opp.get('mtf_confirmation_count', 0),
            opp.get('mtf_total_timeframes', 0),
            opp.get('mtf_confirmation_strength', 0),
            opp.get('priority_boost', 0),
            opp.get('risk_level', 'Unknown'),
            opp.get('chart_file', ''),
            opp.get('signal_type', ''),
            opp.get('distance_from_current', 0),
            0,  # volume_score (placeholder)
            0   # technical_strength (placeholder)
        ]
    
    def append_market_summary(self, results: Dict, scan_id: str) -> bool:
        """Append market summary to the market summary CSV file"""
        try:
            file_path = self.csv_files['market_summary']
            
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
            
            row = [
                scan_id,
                scan_info.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
                scan_info.get('execution_time_seconds', 0),
                scan_info.get('symbols_analyzed', 0),
                scan_info.get('signals_generated', 0),
                scan_info.get('success_rate', 0),
                scan_info.get('charts_generated', 0),
                scan_info.get('parallel_processing', False),
                scan_info.get('threads_used', 0),
                scan_info.get('mtf_enabled', False),
                ', '.join(scan_info.get('confirmation_timeframes', [])),
                self.config.timeframe,
                performance.get('signals_per_minute', 0),
                performance.get('avg_confidence', 0),
                performance.get('avg_original_confidence', 0),
                performance.get('mtf_boost_avg', 0),
                market_summary.get('total_market_volume', 0),
                market_summary.get('average_volume', 0),
                market_summary.get('market_sentiment', {}).get('bullish_percentage', 0),
                market_summary.get('signal_distribution', {}).get('buy_signals', 0),
                market_summary.get('signal_distribution', {}).get('sell_signals', 0),
                market_summary.get('signal_distribution', {}).get('market_orders', 0),
                market_summary.get('signal_distribution', {}).get('limit_orders', 0),
                performance.get('speedup_factor', 1.0),
                mtf_strong,
                mtf_partial,
                mtf_none,
                biggest_gainer.get('symbol', ''),
                biggest_gainer.get('price_change_24h', 0),
                biggest_loser.get('symbol', ''),
                biggest_loser.get('price_change_24h', 0),
                highest_volume.get('symbol', '')
            ]
            
            with open(file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            self.logger.info(f"✅ Appended market summary to {file_path.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error appending market summary: {e}")
            return False
    
    def append_scan_history(self, scan_id: str, results: Dict) -> bool:
        """Append scan information to scan history"""
        try:
            file_path = self.csv_files['scan_history']
            scan_info = results.get('scan_info', {})
            
            # Get system metrics
            import psutil
            memory_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent()
            
            row = [
                scan_id,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                scan_info.get('execution_time_seconds', 0),
                scan_info.get('symbols_analyzed', 0),
                scan_info.get('signals_generated', 0),
                scan_info.get('success_rate', 0),
                scan_info.get('mtf_enabled', False),
                self.config.timeframe,
                ', '.join(self.config.confirmation_timeframes),
                scan_info.get('charts_generated', 0),
                ', '.join([f.name for f in self.csv_files.values()]),
                memory_usage,
                cpu_usage,
                0  # errors_encountered (placeholder)
            ]
            
            with open(file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            self.logger.info(f"✅ Appended scan history to {file_path.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error appending scan history: {e}")
            return False
    
    def append_performance_metrics(self, scan_id: str, metrics: List[Dict]) -> bool:
        """Append performance metrics to the performance CSV file"""
        try:
            if not metrics:
                return True
            
            file_path = self.csv_files['performance_metrics']
            rows_added = 0
            
            with open(file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                for metric in metrics:
                    row = [
                        scan_id,
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        metric.get('type', ''),
                        metric.get('name', ''),
                        metric.get('value', 0),
                        metric.get('unit', ''),
                        metric.get('timeframe', ''),
                        metric.get('symbol', ''),
                        metric.get('additional_data', '')
                    ]
                    writer.writerow(row)
                    rows_added += 1
            
            self.logger.info(f"✅ Appended {rows_added} performance metrics to {file_path.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error appending performance metrics: {e}")
            return False
    
    def export_all_results(self, results: Dict) -> Dict[str, str]:
        """Export all results to CSV files using append mode"""
        try:
            scan_id = self.generate_scan_id()
            updated_files = {}
            
            self.logger.info(f"📊 Exporting results for scan ID: {scan_id}")
            
            # Append signals
            signals = results.get('signals', [])
            if signals and self.append_signals(signals, scan_id):
                updated_files['signals'] = str(self.csv_files['signals'])
            
            # Append opportunities
            opportunities = results.get('top_opportunities', [])
            if opportunities and self.append_opportunities(opportunities, scan_id):
                updated_files['opportunities'] = str(self.csv_files['opportunities'])
            
            # Append market summary
            if self.append_market_summary(results, scan_id):
                updated_files['market_summary'] = str(self.csv_files['market_summary'])
            
            # Append scan history
            if self.append_scan_history(scan_id, results):
                updated_files['scan_history'] = str(self.csv_files['scan_history'])
            
            # Append performance metrics if available
            performance_metrics = results.get('performance_metrics', [])
            if performance_metrics and self.append_performance_metrics(scan_id, performance_metrics):
                updated_files['performance_metrics'] = str(self.csv_files['performance_metrics'])
            
            self.logger.info(f"📄 CSV Export Complete (Scan ID: {scan_id}):")
            for file_type, file_path in updated_files.items():
                try:
                    file_size = os.path.getsize(file_path)
                    file_size_mb = file_size / (1024 * 1024)
                    row_count = self._count_csv_rows(Path(file_path))
                    self.logger.info(f"   {file_type.replace('_', ' ').title()}: {Path(file_path).name} ({file_size_mb:.1f}MB, {row_count} rows)")
                except Exception as e:
                    self.logger.warning(f"   {file_type.replace('_', ' ').title()}: {Path(file_path).name} (size unknown: {e})")
            
            return updated_files
            
        except Exception as e:
            self.logger.error(f"CSV export error: {e}")
            return {}
    
    def _count_csv_rows(self, file_path: Path) -> int:
        """Count rows in CSV file (excluding header)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for line in f) - 1  # Subtract header
        except Exception as e:
            self.logger.warning(f"Error counting rows in {file_path}: {e}")
            return 0
    
    def get_csv_stats(self) -> Dict:
        """Get comprehensive statistics about CSV files"""
        try:
            stats = {
                'summary': {
                    'total_files': len(self.csv_files),
                    'total_size_mb': 0,
                    'total_rows': 0,
                    'directory': str(self.csv_dir)
                },
                'files': {}
            }
            
            for file_type, file_path in self.csv_files.items():
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    row_count = self._count_csv_rows(file_path)
                    last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    file_stats = {
                        'filename': file_path.name,
                        'full_path': str(file_path),
                        'exists': True,
                        'size_bytes': file_size,
                        'size_mb': file_size / (1024 * 1024),
                        'row_count': row_count,
                        'last_modified': last_modified.strftime('%Y-%m-%d %H:%M:%S'),
                        'age_hours': (datetime.now() - last_modified).total_seconds() / 3600
                    }
                    
                    stats['summary']['total_size_mb'] += file_stats['size_mb']
                    stats['summary']['total_rows'] += row_count
                    
                else:
                    file_stats = {
                        'filename': file_path.name,
                        'full_path': str(file_path),
                        'exists': False,
                        'size_bytes': 0,
                        'size_mb': 0,
                        'row_count': 0,
                        'last_modified': 'N/A',
                        'age_hours': 0
                    }
                
                stats['files'][file_type] = file_stats
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting CSV stats: {e}")
            return {'summary': {}, 'files': {}}
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old data from CSV files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cutoff_str = cutoff_date.strftime('%Y-%m-%d')
            cleanup_stats = {}
            
            self.logger.info(f"🧹 Cleaning up data older than {days_to_keep} days (before {cutoff_str})")
            
            for file_type, file_path in self.csv_files.items():
                if not file_path.exists():
                    cleanup_stats[file_type] = 0
                    continue
                
                rows_to_keep = []
                rows_removed = 0
                
                # Read existing data
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if header:
                        rows_to_keep.append(header)
                    
                    for row in reader:
                        if len(row) >= 2:  # Check if timestamp column exists
                            row_date = row[1][:10]  # Extract date part (YYYY-MM-DD)
                            if row_date >= cutoff_str:
                                rows_to_keep.append(row)
                            else:
                                rows_removed += 1
                        else:
                            rows_to_keep.append(row)  # Keep malformed rows
                
                # Write back filtered data if any rows were removed
                if rows_removed > 0:
                    # Create backup before cleaning
                    backup_path = file_path.with_suffix('.backup')
                    file_path.rename(backup_path)
                    
                    with open(file_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerows(rows_to_keep)
                    
                    self.logger.info(f"   {file_type}: removed {rows_removed} old rows, kept {len(rows_to_keep)-1} rows")
                    
                    # Remove backup if successful
                    backup_path.unlink()
                
                cleanup_stats[file_type] = rows_removed
            
            total_removed = sum(cleanup_stats.values())
            self.logger.info(f"🧹 Cleanup complete: removed {total_removed} total rows")
            
            return cleanup_stats
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
            return {}
    
    def backup_csv_files(self, backup_dir: Optional[str] = None) -> bool:
        """Create backup of all CSV files"""
        try:
            if backup_dir is None:
                backup_dir = self.csv_dir / "backups"
            else:
                backup_dir = Path(backup_dir)
            
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_count = 0
            
            for file_type, file_path in self.csv_files.items():
                if file_path.exists():
                    backup_filename = f"{file_path.stem}_{timestamp}{file_path.suffix}"
                    backup_path = backup_dir / backup_filename
                    
                    import shutil
                    shutil.copy2(file_path, backup_path)
                    backup_count += 1
                    
                    self.logger.debug(f"Backed up {file_path.name} → {backup_filename}")
            
            self.logger.info(f"📦 Created backup of {backup_count} CSV files in {backup_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
            return False
    
    def get_recent_scans(self, limit: int = 10) -> List[Dict]:
        """Get information about recent scans"""
        try:
            scan_history_path = self.csv_files['scan_history']
            if not scan_history_path.exists():
                return []
            
            scans = []
            with open(scan_history_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    scans.append({
                        'scan_id': row.get('scan_id', ''),
                        'timestamp': row.get('timestamp', ''),
                        'duration_seconds': float(row.get('duration_seconds', 0)),
                        'symbols_scanned': int(row.get('symbols_scanned', 0)),
                        'signals_found': int(row.get('signals_found', 0)),
                        'success_rate': float(row.get('success_rate', 0)),
                        'charts_generated': int(row.get('charts_generated', 0))
                    })
            
            # Sort by timestamp (most recent first) and limit
            scans.sort(key=lambda x: x['timestamp'], reverse=True)
            return scans[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting recent scans: {e}")
            return []
    
    def export_summary_report(self) -> Optional[str]:
        """Export a summary report of all CSV data"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.csv_dir / f"summary_report_{timestamp}.txt"
            
            stats = self.get_csv_stats()
            recent_scans = self.get_recent_scans()
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("ENHANCED BYBIT TRADING SYSTEM - CSV SUMMARY REPORT\n")
                f.write("=" * 60 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Directory: {stats['summary']['directory']}\n\n")
                
                # Summary statistics
                f.write("SUMMARY STATISTICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total Files: {stats['summary']['total_files']}\n")
                f.write(f"Total Size: {stats['summary']['total_size_mb']:.2f} MB\n")
                f.write(f"Total Rows: {stats['summary']['total_rows']:,}\n\n")
                
                # File details
                f.write("FILE DETAILS\n")
                f.write("-" * 30 + "\n")
                for file_type, file_stats in stats['files'].items():
                    f.write(f"{file_type.replace('_', ' ').title()}:\n")
                    f.write(f"  File: {file_stats['filename']}\n")
                    f.write(f"  Size: {file_stats['size_mb']:.2f} MB\n")
                    f.write(f"  Rows: {file_stats['row_count']:,}\n")
                    f.write(f"  Last Modified: {file_stats['last_modified']}\n")
                    f.write(f"  Age: {file_stats['age_hours']:.1f} hours\n\n")
                
                # Recent scans
                f.write("RECENT SCANS\n")
                f.write("-" * 30 + "\n")
                for scan in recent_scans:
                    f.write(f"Scan ID: {scan['scan_id']}\n")
                    f.write(f"  Time: {scan['timestamp']}\n")
                    f.write(f"  Duration: {scan['duration_seconds']:.1f}s\n")
                    f.write(f"  Symbols: {scan['symbols_scanned']}\n")
                    f.write(f"  Signals: {scan['signals_found']}\n")
                    f.write(f"  Success Rate: {scan['success_rate']:.1f}%\n")
                    f.write(f"  Charts: {scan['charts_generated']}\n\n")
            
            self.logger.info(f"📈 Summary report exported to {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Error exporting summary report: {e}")
            return None