"""
Interactive Chart Generator for the Enhanced Bybit Trading System.
Updated version with clean JPG charts showing only candlesticks and TP/SL boxes.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from datetime import datetime
from typing import Dict, List
from config.config import EnhancedSystemConfig


class InteractiveChartGenerator:
    """Chart generator with TradingView-style charts"""
    
    def __init__(self, config: EnhancedSystemConfig):
        self.config = config
        self.chart_count = 0
        self.logger = logging.getLogger(__name__)
        
    def create_comprehensive_chart(self, symbol: str, df: pd.DataFrame, 
                                 signal_data: Dict, volume_profile: Dict, 
                                 fibonacci_data: Dict, confluence_zones: List[Dict]) -> str:
        """Create TradingView-style chart with signal analysis"""
        try:
            if df.empty:
                return "No data available for charting"
            
            # Create the full interactive chart for browser display
            interactive_fig = self.create_interactive_chart(symbol, df, signal_data, volume_profile, fibonacci_data, confluence_zones)
            
            # Create simplified chart for JPG export
            jpg_fig = self.create_jpg_chart(symbol, df, signal_data)
            
            # Create charts directory if it doesn't exist
            from pathlib import Path
            charts_dir = Path("charts")
            charts_dir.mkdir(exist_ok=True)
            
            # Save JPG version (simplified)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_filename = charts_dir / f"tradingview_{symbol.replace('/', '').replace(':', '')}_chart_{timestamp}.jpg"
            
            jpg_fig.write_image(
                str(chart_filename),
                format='jpg',
                width=self.config.chart_width,
                height=self.config.chart_height,
                scale=2,  # Higher DPI for better quality
                engine='kaleido'  # Use kaleido for better image generation
            )
            
            self.chart_count += 1
            self.logger.info(f"📊 Chart saved: {chart_filename.name}")
            
            # Show interactive chart if configured (HTML version for viewing)
            if self.config.show_charts:
                interactive_fig.show()
            
            return str(chart_filename)
            
        except Exception as e:
            self.logger.error(f"Chart generation error: {e}")
            return f"Chart generation failed: {e}"
    
    def create_interactive_chart(self, symbol: str, df: pd.DataFrame, 
                               signal_data: Dict, volume_profile: Dict, 
                               fibonacci_data: Dict, confluence_zones: List[Dict]):
        """Create full interactive chart with all features"""
        # Create subplots: main chart + Stochastic RSI
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f"{symbol} • Enhanced Analysis with Multi-Timeframe Confirmation", "Stochastic RSI"),
            row_heights=[0.7, 0.3]
        )
        
        # Main candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=symbol,
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350',
                increasing_fillcolor='#26a69a',
                decreasing_fillcolor='#ef5350'
            ),
            row=1, col=1
        )
        
        # Add all technical overlays
        self.add_clean_technical_overlays(fig, df)
        self.add_clean_volume_levels(fig, volume_profile)
        self.add_exact_tpsl_boxes(fig, signal_data, df)
        self.add_clean_signal_markers_with_mtf(fig, signal_data, df)
        
        # Add Stochastic RSI subplot
        fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=df['stoch_rsi_k'], 
                name='Stoch RSI %K', 
                line=dict(color='#2196F3', width=1.5)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=df['stoch_rsi_d'], 
                name='Stoch RSI %D', 
                line=dict(color='#FF9800', width=1.5)
            ),
            row=2, col=1
        )
        
        # Add Stochastic RSI reference lines
        fig.add_hline(y=80, line_dash="dash", line_color="#F44336", row=2, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="#4CAF50", row=2, col=1)
        
        # Clean TradingView-style layout
        fig.update_layout(
            width=1860,
            height=920,
            showlegend=False,
            template="plotly_white",
            font=dict(size=11, color='#131722'),
            margin=dict(l=20, r=80, t=60, b=40),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Update axes
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='#e1e3e6',
            showline=False,
            zeroline=False,
            rangeslider_visible=False,
            tickfont=dict(size=10, color='#787b86'),
            row=1, col=1
        )
        
        fig.update_yaxes(
            title_text="Price",
            showgrid=True,
            gridwidth=1,
            gridcolor='#e1e3e6',
            showline=False,
            zeroline=False,
            side='right',
            tickfont=dict(size=10, color='#787b86'),
            tickformat='.5f' if df['close'].iloc[-1] < 1 else '.2f',
            row=1, col=1
        )
        
        fig.update_yaxes(
            title_text="Stoch RSI",
            range=[0, 100],
            showgrid=True,
            gridwidth=1,
            gridcolor='#e1e3e6',
            row=2, col=1
        )
        
        # Add price information boxes
        self.add_price_boxes_with_mtf(fig, df, signal_data)
        
        return fig
    
    def create_jpg_chart(self, symbol: str, df: pd.DataFrame, signal_data: Dict):
        """Create simplified chart optimized for JPG export - ONLY candlesticks with clear TP/SL boxes"""
        # Create single chart (no subplots for cleaner JPG)
        fig = go.Figure()
        
        # ONLY candlestick chart - no other indicators
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=symbol,
                increasing_line_color='#00C851',
                decreasing_line_color='#FF4444',
                increasing_fillcolor='#00C851',
                decreasing_fillcolor='#FF4444'
            )
        )
        
        # Add ONLY TP/SL boxes - no other technical indicators
        self.add_prominent_tpsl_boxes_for_jpg(fig, signal_data, df)
        
        # Add clear signal marker
        self.add_clear_signal_marker_for_jpg(fig, signal_data, df)
        
        # Clean layout optimized for JPG
        fig.update_layout(
            title=dict(
                text=f"<b>{symbol}</b> • Trading Signal Analysis",
                x=0.5,
                font=dict(size=18, color='#131722', family='Arial Black')
            ),
            width=self.config.chart_width,
            height=self.config.chart_height,
            showlegend=False,
            template="plotly_white",
            font=dict(size=12, color='#131722', family='Arial'),
            margin=dict(l=60, r=60, t=80, b=60),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Update axes for JPG
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='#e1e3e6',
            showline=True,
            linecolor='#131722',
            linewidth=1,
            zeroline=False,
            rangeslider_visible=False,
            tickfont=dict(size=11, color='#131722'),
            title_text="Time",
            title_font=dict(size=12, color='#131722')
        )
        
        fig.update_yaxes(
            title_text="Price (USDT)",
            showgrid=True,
            gridwidth=1,
            gridcolor='#e1e3e6',
            showline=True,
            linecolor='#131722',
            linewidth=1,
            zeroline=False,
            side='right',
            tickfont=dict(size=11, color='#131722'),
            title_font=dict(size=12, color='#131722'),
            tickformat='.5f' if df['close'].iloc[-1] < 1 else '.2f'
        )
        
        return fig
    
    def add_prominent_tpsl_boxes_for_jpg(self, fig, signal_data: Dict, df: pd.DataFrame):
        """Add very prominent and clear TP/SL boxes optimized for JPG"""
        try:
            if not signal_data:
                return
            
            # Extract signal data
            entry_price = signal_data.get('entry_price', 0)
            stop_loss = signal_data.get('stop_loss', 0)
            take_profit_1 = signal_data.get('take_profit_1', 0)
            take_profit_2 = signal_data.get('take_profit_2', 0)
            side = signal_data.get('side', 'buy')
            
            if not all([entry_price, stop_loss, take_profit_1]):
                self.logger.warning("Missing signal data for TP/SL boxes")
                return
            
            # Get time positions for boxes - use the last 40% of the chart
            total_candles = len(df)
            start_pos = int(total_candles * 0.6)  # Start from 60% of chart
            
            if start_pos >= total_candles:
                start_pos = max(0, total_candles - 20)
            
            x_start = df.index[start_pos]
            x_end = df.index[-1]
            
            self.logger.info(f"Adding TP/SL boxes for {side} signal: Entry={entry_price}, SL={stop_loss}, TP1={take_profit_1}, TP2={take_profit_2}")
            
            if side.lower() == 'buy':
                # BUY SIGNAL BOXES
                
                # 1. RED STOP LOSS BOX: stop_loss → entry_price
                fig.add_shape(
                    type="rect",
                    x0=x_start, y0=stop_loss,
                    x1=x_end, y1=entry_price,
                    fillcolor="rgba(255, 68, 68, 0.5)",
                    line=dict(color="#FF4444", width=4),
                    layer="below"
                )
                
                # SL label
                fig.add_annotation(
                    x=x_end, y=(stop_loss + entry_price) / 2,
                    text=f"<b>STOP LOSS</b><br>${stop_loss:.5f}",
                    showarrow=False,
                    font=dict(size=16, color="#FF4444", family="Arial Black"),
                    bgcolor="rgba(255,255,255,0.95)", 
                    bordercolor="#FF4444", borderwidth=3,
                    xshift=10
                )
                
                # 2. GREEN TP1 BOX: entry_price → take_profit_1
                fig.add_shape(
                    type="rect",
                    x0=x_start, y0=entry_price,
                    x1=x_end, y1=take_profit_1,
                    fillcolor="rgba(76, 175, 80, 0.5)",
                    line=dict(color="#4CAF50", width=4),
                    layer="below"
                )
                
                # TP1 label
                fig.add_annotation(
                    x=x_end, y=(entry_price + take_profit_1) / 2,
                    text=f"<b>TAKE PROFIT 1</b><br>${take_profit_1:.5f}",
                    showarrow=False,
                    font=dict(size=16, color="#4CAF50", family="Arial Black"),
                    bgcolor="rgba(255,255,255,0.95)", 
                    bordercolor="#4CAF50", borderwidth=3,
                    xshift=10
                )
                
                # 3. DARK GREEN TP2 BOX (if exists)
                if take_profit_2 > 0 and take_profit_2 > take_profit_1:
                    fig.add_shape(
                        type="rect",
                        x0=x_start, y0=take_profit_1,
                        x1=x_end, y1=take_profit_2,
                        fillcolor="rgba(27, 94, 32, 0.5)",
                        line=dict(color="#1B5E20", width=4),
                        layer="below"
                    )
                    
                    # TP2 label
                    fig.add_annotation(
                        x=x_end, y=(take_profit_1 + take_profit_2) / 2,
                        text=f"<b>TAKE PROFIT 2</b><br>${take_profit_2:.5f}",
                        showarrow=False,
                        font=dict(size=16, color="#1B5E20", family="Arial Black"),
                        bgcolor="rgba(255,255,255,0.95)", 
                        bordercolor="#1B5E20", borderwidth=3,
                        xshift=10
                    )
            
            else:  # SELL SIGNAL
                # SELL SIGNAL BOXES
                
                # 1. RED STOP LOSS BOX: entry_price → stop_loss
                fig.add_shape(
                    type="rect",
                    x0=x_start, y0=entry_price,
                    x1=x_end, y1=stop_loss,
                    fillcolor="rgba(255, 68, 68, 0.5)",
                    line=dict(color="#FF4444", width=4),
                    layer="below"
                )
                
                # SL label
                fig.add_annotation(
                    x=x_end, y=(entry_price + stop_loss) / 2,
                    text=f"<b>STOP LOSS</b><br>${stop_loss:.5f}",
                    showarrow=False,
                    font=dict(size=16, color="#FF4444", family="Arial Black"),
                    bgcolor="rgba(255,255,255,0.95)", 
                    bordercolor="#FF4444", borderwidth=3,
                    xshift=10
                )
                
                # 2. GREEN TP1 BOX: take_profit_1 → entry_price
                fig.add_shape(
                    type="rect",
                    x0=x_start, y0=take_profit_1,
                    x1=x_end, y1=entry_price,
                    fillcolor="rgba(76, 175, 80, 0.5)",
                    line=dict(color="#4CAF50", width=4),
                    layer="below"
                )
                
                # TP1 label
                fig.add_annotation(
                    x=x_end, y=(take_profit_1 + entry_price) / 2,
                    text=f"<b>TAKE PROFIT 1</b><br>${take_profit_1:.5f}",
                    showarrow=False,
                    font=dict(size=16, color="#4CAF50", family="Arial Black"),
                    bgcolor="rgba(255,255,255,0.95)", 
                    bordercolor="#4CAF50", borderwidth=3,
                    xshift=10
                )
                
                # 3. DARK GREEN TP2 BOX (if exists)
                if take_profit_2 > 0 and take_profit_2 < take_profit_1:
                    fig.add_shape(
                        type="rect",
                        x0=x_start, y0=take_profit_2,
                        x1=x_end, y1=take_profit_1,
                        fillcolor="rgba(27, 94, 32, 0.5)",
                        line=dict(color="#1B5E20", width=4),
                        layer="below"
                    )
                    
                    # TP2 label
                    fig.add_annotation(
                        x=x_end, y=(take_profit_2 + take_profit_1) / 2,
                        text=f"<b>TAKE PROFIT 2</b><br>${take_profit_2:.5f}",
                        showarrow=False,
                        font=dict(size=16, color="#1B5E20", family="Arial Black"),
                        bgcolor="rgba(255,255,255,0.95)", 
                        bordercolor="#1B5E20", borderwidth=3,
                        xshift=10
                    )
            
            # Add prominent entry price line across the entire chart
            fig.add_hline(
                y=entry_price,
                line_dash="solid",
                line_color="#131722",
                line_width=3,
                annotation_text=f"<b>ENTRY: ${entry_price:.5f}</b>",
                annotation_position="middle left",
                annotation_font=dict(size=16, color="#131722", family="Arial Black"),
                annotation_bgcolor="rgba(255,255,255,0.95)",
                annotation_bordercolor="#131722",
                annotation_borderwidth=3
            )
            
            self.logger.info(f"✅ Successfully added TP/SL boxes for {side} signal")
                
        except Exception as e:
            self.logger.error(f"Error adding prominent TP/SL boxes: {e}")
            import traceback
            traceback.print_exc()
    
    def add_clear_signal_marker_for_jpg(self, fig, signal_data: Dict, df: pd.DataFrame):
        """Add clear, prominent signal marker for JPG"""
        try:
            if not signal_data:
                return
            
            current_price = df['close'].iloc[-1]
            entry_price = signal_data.get('entry_price', current_price)
            side = signal_data.get('side', 'buy')
            confidence = signal_data.get('confidence', 0)
            
            # Get MTF confirmation data
            mtf_analysis = signal_data.get('mtf_analysis', {})
            confirmed_timeframes = mtf_analysis.get('confirmed_timeframes', [])
            
            # Choose marker based on side and MTF status
            if side.lower() == 'buy':
                marker_color = '#00C851'
                marker_symbol = 'triangle-up'
                signal_text = f"<b>🟢 BUY SIGNAL</b>"
            else:
                marker_color = '#FF4444'
                marker_symbol = 'triangle-down'
                signal_text = f"<b>🔴 SELL SIGNAL</b>"
            
            # Add large, prominent signal marker
            fig.add_trace(
                go.Scatter(
                    x=[df.index[-1]],
                    y=[entry_price],
                    mode='markers',
                    marker=dict(
                        symbol=marker_symbol,
                        size=25,  # Large marker
                        color=marker_color,
                        line=dict(width=3, color='white')
                    ),
                    showlegend=False
                )
            )
            
            # Add signal information box
            total_timeframes = len(signal_data.get('config', {}).get('confirmation_timeframes', []))
            if total_timeframes == 0:
                total_timeframes = 2  # Default assumption
            
            mtf_status = f"MTF: {len(confirmed_timeframes)}/{total_timeframes}"
            
            fig.add_annotation(
                x=df.index[-10] if len(df) >= 10 else df.index[0],
                y=entry_price,
                text=f"{signal_text}<br>Confidence: {confidence:.0f}%<br>{mtf_status}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowcolor=marker_color,
                ax=0, ay=-80,
                font=dict(size=14, color=marker_color, family="Arial Black"),
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor=marker_color,
                borderwidth=2
            )
                
        except Exception as e:
            self.logger.error(f"Error adding clear signal marker: {e}")
    
    def add_exact_tpsl_boxes(self, fig, signal_data: Dict, df: pd.DataFrame):
        """Add boxes with exact price boundaries for interactive chart"""
        try:
            if not signal_data:
                return
            
            # Extract signal data
            entry_price = signal_data.get('entry_price', 0)
            stop_loss = signal_data.get('stop_loss', 0)
            take_profit_1 = signal_data.get('take_profit_1', 0)
            take_profit_2 = signal_data.get('take_profit_2', 0)
            side = signal_data.get('side', 'buy')
            
            if not all([entry_price, stop_loss, take_profit_1]):
                return
            
            # Get time range for boxes
            x_start = df.index[-20] if len(df) >= 20 else df.index[0]
            x_end = df.index[-1]
            time_diff = df.index[-1] - df.index[-2] if len(df) > 1 else pd.Timedelta(hours=1)
            x_future = x_end + (time_diff * 10)
            
            if side.lower() == 'buy':
                # BUY SIGNAL BOXES
                
                # 1. RED STOP LOSS BOX: stop_loss → entry_price
                if stop_loss > 0:
                    fig.add_shape(
                        type="rect",
                        x0=x_start, y0=stop_loss,
                        x1=x_future, y1=entry_price,
                        fillcolor="rgba(244, 67, 54, 0.25)",
                        line=dict(color="rgba(244, 67, 54, 0.8)", width=2),
                        layer="below",
                        row=1, col=1
                    )
                    
                    # SL annotation
                    fig.add_annotation(
                        x=x_end, y=(stop_loss + entry_price) / 2,
                        text="SL", showarrow=True, arrowhead=2,
                        arrowcolor="#F44336", ax=60, ay=0,
                        font=dict(size=12, color="#F44336", family="Arial Black"),
                        bgcolor="rgba(255,255,255,0.9)", 
                        bordercolor="#F44336", borderwidth=1,
                        row=1, col=1
                    )
                
                # 2. GREEN TP1 BOX: entry_price → take_profit_1
                if take_profit_1 > 0:
                    fig.add_shape(
                        type="rect",
                        x0=x_start, y0=entry_price,
                        x1=x_future, y1=take_profit_1,
                        fillcolor="rgba(76, 175, 80, 0.25)",
                        line=dict(color="rgba(76, 175, 80, 0.8)", width=2),
                        layer="below",
                        row=1, col=1
                    )
                    
                    # TP1 annotation
                    fig.add_annotation(
                        x=x_end, y=(entry_price + take_profit_1) / 2,
                        text="TP1", showarrow=True, arrowhead=2,
                        arrowcolor="#4CAF50", ax=80, ay=0,
                        font=dict(size=12, color="#4CAF50", family="Arial Black"),
                        bgcolor="rgba(255,255,255,0.9)", 
                        bordercolor="#4CAF50", borderwidth=1,
                        row=1, col=1
                    )
                
                # 3. DARK GREEN TP2 BOX: take_profit_1 → take_profit_2
                if take_profit_2 > 0 and take_profit_2 > take_profit_1:
                    fig.add_shape(
                        type="rect",
                        x0=x_start, y0=take_profit_1,
                        x1=x_future, y1=take_profit_2,
                        fillcolor="rgba(27, 94, 32, 0.25)",
                        line=dict(color="rgba(27, 94, 32, 0.8)", width=2),
                        layer="below",
                        row=1, col=1
                    )
                    
                    # TP2 annotation
                    fig.add_annotation(
                        x=x_end, y=(take_profit_1 + take_profit_2) / 2,
                        text="TP2", showarrow=True, arrowhead=2,
                        arrowcolor="#1B5E20", ax=100, ay=0,
                        font=dict(size=12, color="#1B5E20", family="Arial Black"),
                        bgcolor="rgba(255,255,255,0.9)", 
                        bordercolor="#1B5E20", borderwidth=1,
                        row=1, col=1
                    )
            
            else:  # SELL SIGNAL
                # SELL SIGNAL BOXES
                
                # 1. RED STOP LOSS BOX: entry_price → stop_loss
                if stop_loss > 0:
                    fig.add_shape(
                        type="rect",
                        x0=x_start, y0=entry_price,
                        x1=x_future, y1=stop_loss,
                        fillcolor="rgba(244, 67, 54, 0.25)",
                        line=dict(color="rgba(244, 67, 54, 0.8)", width=2),
                        layer="below",
                        row=1, col=1
                    )
                    
                    # SL annotation
                    fig.add_annotation(
                        x=x_end, y=(entry_price + stop_loss) / 2,
                        text="SL", showarrow=True, arrowhead=2,
                        arrowcolor="#F44336", ax=60, ay=0,
                        font=dict(size=12, color="#F44336", family="Arial Black"),
                        bgcolor="rgba(255,255,255,0.9)", 
                        bordercolor="#F44336", borderwidth=1,
                        row=1, col=1
                    )
                
                # 2. GREEN TP1 BOX: take_profit_1 → entry_price
                if take_profit_1 > 0:
                    fig.add_shape(
                        type="rect",
                        x0=x_start, y0=take_profit_1,
                        x1=x_future, y1=entry_price,
                        fillcolor="rgba(76, 175, 80, 0.25)",
                        line=dict(color="rgba(76, 175, 80, 0.8)", width=2),
                        layer="below",
                        row=1, col=1
                    )
                    
                    # TP1 annotation
                    fig.add_annotation(
                        x=x_end, y=(take_profit_1 + entry_price) / 2,
                        text="TP1", showarrow=True, arrowhead=2,
                        arrowcolor="#4CAF50", ax=80, ay=0,
                        font=dict(size=12, color="#4CAF50", family="Arial Black"),
                        bgcolor="rgba(255,255,255,0.9)", 
                        bordercolor="#4CAF50", borderwidth=1,
                        row=1, col=1
                    )
                
                # 3. DARK GREEN TP2 BOX: take_profit_2 → take_profit_1
                if take_profit_2 > 0 and take_profit_2 < take_profit_1:
                    fig.add_shape(
                        type="rect",
                        x0=x_start, y0=take_profit_2,
                        x1=x_future, y1=take_profit_1,
                        fillcolor="rgba(27, 94, 32, 0.25)",
                        line=dict(color="rgba(27, 94, 32, 0.8)", width=2),
                        layer="below",
                        row=1, col=1
                    )
                    
                    # TP2 annotation
                    fig.add_annotation(
                        x=x_end, y=(take_profit_2 + take_profit_1) / 2,
                        text="TP2", showarrow=True, arrowhead=2,
                        arrowcolor="#1B5E20", ax=100, ay=0,
                        font=dict(size=12, color="#1B5E20", family="Arial Black"),
                        bgcolor="rgba(255,255,255,0.9)", 
                        bordercolor="#1B5E20", borderwidth=1,
                        row=1, col=1
                    )
            
            # Add entry price line
            fig.add_hline(
                y=entry_price,
                line_dash="solid",
                line_color="#607D8B",
                line_width=2,
                annotation_text=f"Entry: {entry_price:.5f}",
                annotation_position="bottom left",
                annotation_font_color="#607D8B",
                row=1, col=1
            )
                
        except Exception as e:
            self.logger.error(f"Error adding exact TP/SL boxes: {e}")
    
    def add_clean_technical_overlays(self, fig, df: pd.DataFrame):
        """Add clean technical overlays including Ichimoku"""
        try:
            # Moving averages with TradingView colors
            if 'sma_20' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df['sma_20'], mode='lines', name='SMA 20',
                        line=dict(color='#2196f3', width=1.5),
                        hovertemplate='SMA 20: %{y:.5f}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            if 'sma_50' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df['sma_50'], mode='lines', name='SMA 50',
                        line=dict(color='#ff9800', width=1.5),
                        hovertemplate='SMA 50: %{y:.5f}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # VWAP
            if 'vwap' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df['vwap'], mode='lines', name='VWAP',
                        line=dict(color='#9c27b0', width=2),
                        hovertemplate='VWAP: %{y:.5f}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # Ichimoku Cloud
            if all(col in df.columns for col in ['ichimoku_span_a', 'ichimoku_span_b', 'ichimoku_tenkan', 'ichimoku_kijun']):
                # Ichimoku Cloud (Senkou Spans)
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df['ichimoku_span_a'], 
                        name='Senkou Span A', 
                        line=dict(color='#26A69A', width=1), 
                        opacity=0.3
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df['ichimoku_span_b'], 
                        name='Senkou Span B', 
                        line=dict(color='#EF5350', width=1), 
                        opacity=0.3,
                        fill='tonexty', 
                        fillcolor='rgba(100, 100, 100, 0.2)'
                    ),
                    row=1, col=1
                )
                
                # Tenkan-sen and Kijun-sen
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df['ichimoku_tenkan'], 
                        name='Tenkan-sen', 
                        line=dict(color='#0288D1', width=1.5)
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df['ichimoku_kijun'], 
                        name='Kijun-sen', 
                        line=dict(color='#D81B60', width=1.5)
                    ),
                    row=1, col=1
                )
            
            # Support and Resistance as clean lines
            if 'support' in df.columns and 'resistance' in df.columns:
                latest_support = df['support'].dropna().iloc[-1] if not df['support'].dropna().empty else 0
                latest_resistance = df['resistance'].dropna().iloc[-1] if not df['resistance'].dropna().empty else 0
                
                if latest_support > 0:
                    fig.add_hline(
                        y=latest_support, line_dash="dash", line_color="#4caf50", line_width=1,
                        annotation_text="Support", annotation_position="bottom right",
                        row=1, col=1
                    )
                
                if latest_resistance > 0:
                    fig.add_hline(
                        y=latest_resistance, line_dash="dash", line_color="#f44336", line_width=1,
                        annotation_text="Resistance", annotation_position="top right",
                        row=1, col=1
                    )
                    
        except Exception as e:
            self.logger.error(f"Error adding clean technical overlays: {e}")
    
    def add_clean_volume_levels(self, fig, volume_profile: Dict):
        """Add volume profile levels as clean lines"""
        try:
            if not volume_profile:
                return
            
            # POC - Point of Control (most important)
            if volume_profile.get('poc', 0) > 0:
                fig.add_hline(
                    y=volume_profile['poc'], line_dash="solid", line_color="#ffc107", line_width=3,
                    annotation_text="POC", annotation_position="bottom right", annotation_font_color="#ffc107",
                    row=1, col=1
                )
            
            # Value Area High and Low
            if volume_profile.get('vah', 0) > 0:
                fig.add_hline(
                    y=volume_profile['vah'], line_dash="dot", line_color="#ff9800", line_width=1,
                    annotation_text="VAH", annotation_position="bottom right",
                    row=1, col=1
                )
            
            if volume_profile.get('val', 0) > 0:
                fig.add_hline(
                    y=volume_profile['val'], line_dash="dot", line_color="#ff9800", line_width=1,
                    annotation_text="VAL", annotation_position="top right",
                    row=1, col=1
                )
            
            # Value Area as subtle rectangle
            if volume_profile.get('vah', 0) > 0 and volume_profile.get('val', 0) > 0:
                fig.add_hrect(
                    y0=volume_profile['val'], y1=volume_profile['vah'],
                    fillcolor="rgba(255,193,7,0.1)", layer="below", line_width=0,
                    row=1, col=1
                )
                
        except Exception as e:
            self.logger.error(f"Error adding volume levels: {e}")
    
    def add_clean_signal_markers_with_mtf(self, fig, signal_data: Dict, df: pd.DataFrame):
        """Add clean signal markers with MTF confirmation information"""
        try:
            if not signal_data:
                return
            
            current_price = df['close'].iloc[-1]
            entry_price = signal_data.get('entry_price', current_price)
            side = signal_data.get('side', 'buy')
            
            # Get MTF confirmation data
            mtf_analysis = signal_data.get('mtf_analysis', {})
            confirmed_timeframes = mtf_analysis.get('confirmed_timeframes', [])
            
            # Choose marker style based on MTF confirmation
            if len(confirmed_timeframes) >= 2:
                marker_color = '#1e88e5' if side == 'buy' else '#e53935'
                marker_size = 15
                marker_symbol = 'star' if side == 'buy' else 'star'
            elif len(confirmed_timeframes) >= 1:
                marker_color = '#26a69a' if side == 'buy' else '#ef5350'
                marker_size = 12
                marker_symbol = 'triangle-up' if side == 'buy' else 'triangle-down'
            else:
                marker_color = '#757575'
                marker_size = 10
                marker_symbol = 'circle'
            
            # Get confirmation timeframes count
            config_timeframes = getattr(self.config, 'confirmation_timeframes', [])
            total_timeframes = len(config_timeframes)
            
            fig.add_trace(
                go.Scatter(
                    x=[df.index[-1]],
                    y=[entry_price],
                    mode='markers',
                    marker=dict(
                        symbol=marker_symbol,
                        size=marker_size,
                        color=marker_color,
                        line=dict(width=2, color='white')
                    ),
                    name=f"{side.upper()} Signal (MTF: {len(confirmed_timeframes)}/{total_timeframes})",
                    hovertemplate=f'{side.upper()} Signal<br>Entry: %{{y:.5f}}<br>MTF Confirmation: {len(confirmed_timeframes)}/{total_timeframes}<extra></extra>'
                ),
                row=1, col=1
            )
                
        except Exception as e:
            self.logger.error(f"Error adding MTF signal markers: {e}")
    
    def add_price_boxes_with_mtf(self, fig, df: pd.DataFrame, signal_data: Dict):
        """Add price information boxes with MTF confirmation data"""
        try:
            current_price = df['close'].iloc[-1]
            price_change = df['close'].iloc[-1] - df['close'].iloc[-2] if len(df) > 1 else 0
            price_change_pct = (price_change / df['close'].iloc[-2] * 100) if len(df) > 1 and df['close'].iloc[-2] != 0 else 0
            
            # Price info text
            color = '#26a69a' if price_change >= 0 else '#ef5350'
            change_symbol = '+' if price_change >= 0 else ''
            
            price_text = (
                f"<b>{current_price:.5f}</b><br>"
                f"<span style='color:{color}'>{change_symbol}{price_change:.5f} ({change_symbol}{price_change_pct:.2f}%)</span>"
            )
            
            # Add price box in top-left corner
            fig.add_annotation(
                text=price_text,
                x=0.02, y=0.98,
                xref="paper", yref="paper",
                xanchor="left", yanchor="top",
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1,
                font=dict(size=12, color="#131722")
            )
            
            # Enhanced signal info box with MTF data
            if signal_data:
                mtf_analysis = signal_data.get('mtf_analysis', {})
                confirmed_timeframes = mtf_analysis.get('confirmed_timeframes', [])
                original_confidence = signal_data.get('original_confidence', signal_data.get('confidence', 0))
                boosted_confidence = signal_data.get('confidence', 0)
                
                # Choose color based on MTF confirmation
                if len(confirmed_timeframes) >= 2:
                    signal_color = '#1e88e5' if signal_data.get('side') == 'buy' else '#e53935'
                    mtf_status = "STRONG"
                elif len(confirmed_timeframes) >= 1:
                    signal_color = '#26a69a' if signal_data.get('side') == 'buy' else '#ef5350'
                    mtf_status = "PARTIAL"
                else:
                    signal_color = '#757575'
                    mtf_status = "NONE"
                
                # Build signal text with TP1 and TP2
                tp1 = signal_data.get('take_profit_1', 0)
                tp2 = signal_data.get('take_profit_2', 0)
                
                signal_text = (
                    f"<b>{signal_data.get('side', '').upper()} SIGNAL</b><br>"
                    f"Entry: {signal_data.get('entry_price', 0):.5f}<br>"
                    f"Stop Loss: {signal_data.get('stop_loss', 0):.5f}<br>"
                    f"TP1: {tp1:.5f}<br>"
                )
                
                # Add TP2 if it exists and is different from TP1
                if tp2 > 0 and abs(tp2 - tp1) > tp1 * 0.01:
                    signal_text += f"TP2: {tp2:.5f}<br>"
                
                # Get confirmation timeframes count
                config_timeframes = getattr(self.config, 'confirmation_timeframes', [])
                total_timeframes = len(config_timeframes)
                
                signal_text += (
                    f"Confidence: {boosted_confidence:.1f}% ({original_confidence:.1f}% + MTF)<br>"
                    f"MTF Status: {mtf_status} ({len(confirmed_timeframes)}/{total_timeframes})<br>"
                    f"Confirmed: {', '.join(confirmed_timeframes) if confirmed_timeframes else 'None'}<br>"
                    f"Type: {signal_data.get('order_type', '').upper()}"
                )
                
                # Positioned at bottom-right
                fig.add_annotation(
                    text=signal_text,
                    x=0.99, y=0.02,
                    xref="paper", yref="paper", 
                    xanchor="right", yanchor="bottom",
                    bgcolor=f"rgba{(*[int(signal_color[i:i+2], 16) for i in (1, 3, 5)], 0.1)}",
                    bordercolor=signal_color,
                    borderwidth=2,
                    font=dict(size=10, color=signal_color)
                )
                
        except Exception as e:
            self.logger.error(f"Error adding MTF price boxes: {e}")