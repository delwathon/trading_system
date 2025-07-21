"""
Interactive Chart Generator for the Enhanced Bybit Trading System.
Updated version with browser screenshot capture instead of JPG export.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import time
import os
from datetime import datetime
from typing import Dict, List
from pathlib import Path
from config.config import EnhancedSystemConfig


class InteractiveChartGenerator:
    """Chart generator with TradingView-style charts and automatic screenshot capture"""
    
    def __init__(self, config: EnhancedSystemConfig):
        self.config = config
        self.chart_count = 0
        self.logger = logging.getLogger(__name__)
        
        # Create charts directory
        self.charts_dir = Path("charts")
        self.charts_dir.mkdir(exist_ok=True)
        
    def create_comprehensive_chart(self, symbol: str, df: pd.DataFrame, 
                                 signal_data: Dict, volume_profile: Dict, 
                                 fibonacci_data: Dict, confluence_zones: List[Dict]) -> str:
        """Create TradingView-style chart with automatic screenshot capture"""
        try:
            if df.empty:
                return "No data available for charting"
            
            # Create the interactive chart
            fig = self.create_interactive_chart(symbol, df, signal_data, volume_profile, fibonacci_data, confluence_zones)
            
            # Generate timestamp and filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_symbol = symbol.replace('/', '').replace(':', '')
            
            # Option 1: Save HTML file (set to False if you don't want HTML files)
            save_html = getattr(self.config, 'save_charts', False)  # Default: True, set to False to disable
            
            if save_html:
                # Save HTML version
                html_filename = self.charts_dir / f"tradingview_{clean_symbol}_chart_{timestamp}.html"
                fig.write_html(
                    str(html_filename),
                    include_plotlyjs='cdn',  # Use CDN for smaller file size
                    config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': f'{symbol}_trading_signal',
                            'height': 1200,
                            'width': 1800,
                            'scale': 2
                        }
                    }
                )
                
                # Take screenshot of the HTML chart
                screenshot_path = self.capture_chart_screenshot(html_filename, symbol)
                
                self.logger.debug(f"📊 Chart saved: HTML={html_filename.name}, Screenshot={os.path.basename(screenshot_path) if screenshot_path else 'Failed'}")
            else:
                # Option 2: Create temporary HTML file just for screenshot, then delete it
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as temp_file:
                    fig.write_html(
                        temp_file.name,
                        include_plotlyjs='cdn',
                        config={'displayModeBar': False}
                    )
                    temp_html_path = Path(temp_file.name)
                
                # Take screenshot of temporary HTML
                screenshot_path = self.capture_chart_screenshot(temp_html_path, symbol)
                
                # Delete temporary HTML file
                try:
                    temp_html_path.unlink()
                except:
                    pass
                
                self.logger.debug(f"📊 Chart screenshot saved: {os.path.basename(screenshot_path) if screenshot_path else 'Failed'} (no HTML saved)")
            
            self.chart_count += 1
            
            # Show interactive chart if configured
            if self.config.show_charts:
                try:
                    fig.show()
                except Exception as e:
                    self.logger.warning(f"Could not display chart in browser: {e}")
            
            # Return the screenshot path (or temp path if screenshot failed)
            return screenshot_path if screenshot_path else f"Chart generated for {symbol}"
            
        except Exception as e:
            self.logger.error(f"Chart generation error: {e}")
            return f"Chart generation failed: {e}"
    
    def capture_chart_screenshot(self, html_path: Path, symbol: str) -> str:
        """Capture screenshot of HTML chart using Selenium"""
        try:
            # Try Selenium first (more reliable)
            screenshot_path = self._capture_with_selenium(html_path, symbol)
            if screenshot_path:
                return screenshot_path
            
            # Fallback to Playwright
            screenshot_path = self._capture_with_playwright(html_path, symbol)
            if screenshot_path:
                return screenshot_path
            
            self.logger.warning("Screenshot capture failed, returning HTML path")
            return str(html_path)
            
        except Exception as e:
            self.logger.error(f"Screenshot capture error: {e}")
            return str(html_path)
    
    def _capture_with_selenium(self, html_path: Path, symbol: str) -> str:
        """Capture screenshot using Selenium WebDriver"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            
            # Setup Chrome options for full page capture
            chrome_options = Options()
            chrome_options.add_argument('--headless')  # Run in background
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--disable-web-security')
            chrome_options.add_argument('--allow-running-insecure-content')
            chrome_options.add_argument('--hide-scrollbars')
            chrome_options.add_argument('--disable-extensions')
            # Set a large window size to ensure full chart visibility
            chrome_options.add_argument(f'--window-size={max(1920, self.config.chart_width)},{max(1200, self.config.chart_height)}')
            
            # Initialize driver
            service = Service()  # Uses system PATH to find chromedriver
            driver = webdriver.Chrome(service=service, options=chrome_options)
            
            try:
                # Set window size again to be sure
                driver.set_window_size(max(1920, self.config.chart_width), max(1200, self.config.chart_height))
                
                # Load the HTML file
                file_url = f"file://{html_path.absolute()}"
                driver.get(file_url)
                
                # Wait for Plotly chart to load
                wait = WebDriverWait(driver, 30)
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "plotly-graph-div")))
                
                # Wait for all chart elements to load
                time.sleep(5)
                
                # Execute JavaScript to ensure chart is fully rendered
                driver.execute_script("window.scrollTo(0, 0);")  # Scroll to top
                time.sleep(2)
                
                # Try to find the chart container and get its size
                try:
                    chart_element = driver.find_element(By.CLASS_NAME, "plotly-graph-div")
                    chart_size = chart_element.size
                    chart_location = chart_element.location
                    
                    self.logger.debug(f"Chart element size: {chart_size}, location: {chart_location}")
                    
                    # Set window size based on chart size if available
                    if chart_size['width'] > 0 and chart_size['height'] > 0:
                        # Add padding to ensure full visibility
                        window_width = max(1920, chart_size['width'] + 100)
                        window_height = max(1200, chart_size['height'] + 200)
                        driver.set_window_size(window_width, window_height)
                        time.sleep(2)
                        
                except Exception as e:
                    self.logger.warning(f"Could not get chart element size: {e}")
                
                # Take full page screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_filename = f"tradingview_{symbol.replace('/', '').replace(':', '')}_screenshot_{timestamp}.png"
                screenshot_path = self.charts_dir / screenshot_filename
                
                # Get page height and take full page screenshot
                total_height = driver.execute_script("return Math.max( document.body.scrollHeight, document.body.offsetHeight, document.documentElement.clientHeight, document.documentElement.scrollHeight, document.documentElement.offsetHeight );")
                
                # Set window height to full page height
                driver.set_window_size(max(1920, self.config.chart_width), total_height + 100)
                time.sleep(2)
                
                # Take screenshot of full page
                driver.save_screenshot(str(screenshot_path))
                
                self.logger.debug(f"✅ Screenshot captured with Selenium: {screenshot_filename} (Full page: {total_height}px)")
                return str(screenshot_path)
                
            finally:
                driver.quit()
                
        except ImportError:
            self.logger.warning("Selenium not installed. Install with: pip install selenium")
            return None
        except Exception as e:
            self.logger.warning(f"Selenium screenshot failed: {e}")
            return None
    
    def _capture_with_playwright(self, html_path: Path, symbol: str) -> str:
        """Capture screenshot using Playwright (fallback)"""
        try:
            from playwright.sync_api import sync_playwright
            
            with sync_playwright() as p:
                # Launch browser with larger viewport
                browser = p.chromium.launch(headless=True)
                
                # Create page with large viewport
                page = browser.new_page(viewport={
                    'width': max(1920, self.config.chart_width), 
                    'height': max(1200, self.config.chart_height)
                })
                
                # Load HTML file
                file_url = f"file://{html_path.absolute()}"
                page.goto(file_url, wait_until='networkidle')
                
                # Wait for Plotly chart to load
                page.wait_for_selector('.plotly-graph-div', timeout=30000)
                
                # Additional wait for chart rendering and animations
                page.wait_for_timeout(5000)
                
                # Scroll to ensure everything is visible
                page.evaluate("window.scrollTo(0, 0)")
                page.wait_for_timeout(1000)
                
                # Get page dimensions
                page_height = page.evaluate("() => Math.max( document.body.scrollHeight, document.body.offsetHeight, document.documentElement.clientHeight, document.documentElement.scrollHeight, document.documentElement.offsetHeight )")
                
                # Set viewport to full page height
                page.set_viewport_size(max(1920, self.config.chart_width), page_height + 100)
                page.wait_for_timeout(2000)
                
                # Take screenshot of full page
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_filename = f"tradingview_{symbol.replace('/', '').replace(':', '')}_screenshot_{timestamp}.png"
                screenshot_path = self.charts_dir / screenshot_filename
                
                page.screenshot(path=str(screenshot_path), full_page=True)
                
                browser.close()
                
                self.logger.debug(f"✅ Screenshot captured with Playwright: {screenshot_filename} (Full page: {page_height}px)")
                return str(screenshot_path)
                
        except ImportError:
            self.logger.warning("Playwright not installed. Install with: pip install playwright")
            return None
        except Exception as e:
            self.logger.warning(f"Playwright screenshot failed: {e}")
            return None
    
    def create_interactive_chart(self, symbol: str, df: pd.DataFrame, 
                               signal_data: Dict, volume_profile: Dict, 
                               fibonacci_data: Dict, confluence_zones: List[Dict]):
        """Create comprehensive interactive chart with all features"""
        # Create subplots: main chart + volume + Stochastic RSI
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(
                f"{symbol} • Enhanced Multi-Timeframe Analysis", 
                "Volume Profile", 
                "Stochastic RSI"
            ),
            row_heights=[0.6, 0.2, 0.2]
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
                increasing_fillcolor='rgba(38, 166, 154, 0.8)',
                decreasing_fillcolor='rgba(239, 83, 80, 0.8)'
            ),
            row=1, col=1
        )
        
        # Add technical overlays to main chart
        self.add_technical_overlays(fig, df)
        self.add_volume_levels(fig, volume_profile)
        self.add_fibonacci_levels(fig, fibonacci_data)
        self.add_confluence_zones(fig, confluence_zones)
        self.add_signal_annotations(fig, signal_data, df)
        
        # Volume profile in second subplot
        if 'volume' in df.columns:
            colors = ['#26a69a' if close >= open_price else '#ef5350' 
                     for close, open_price in zip(df['close'], df['open'])]
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Stochastic RSI in third subplot
        if 'stoch_rsi_k' in df.columns and 'stoch_rsi_d' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df['stoch_rsi_k'], 
                    name='Stoch RSI %K', 
                    line=dict(color='#2196F3', width=2)
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df['stoch_rsi_d'], 
                    name='Stoch RSI %D', 
                    line=dict(color='#FF9800', width=2)
                ),
                row=3, col=1
            )
            
            # Add reference lines
            fig.add_hline(y=80, line_dash="dash", line_color="#F44336", opacity=0.7, row=3, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="#4CAF50", opacity=0.7, row=3, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="#666", opacity=0.5, row=3, col=1)
        
        # Professional TradingView-style layout
        fig.update_layout(
            title=dict(
                text=f"<b>{symbol}</b> • Multi-Timeframe Trading Signal Analysis",
                x=0.5,
                font=dict(size=24, color='#2c3e50', family='Arial Black')
            ),
            width=1800,  # Larger width for better visibility
            height=1200,  # Larger height for full 3-panel view
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.05,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
                font=dict(size=11)
            ),
            template="plotly_white",
            font=dict(size=12, color='#2c3e50', family='Arial'),
            margin=dict(l=60, r=60, t=100, b=80),  # Increased margins
            plot_bgcolor='#fafafa',
            paper_bgcolor='white'
        )
        
        # Update axes styling
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)',
            showline=True,
            linewidth=1,
            linecolor='rgba(0,0,0,0.2)',
            mirror=True,
            tickfont=dict(size=10, color='#2c3e50'),
            rangeslider_visible=False
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)',
            showline=True,
            linewidth=1,
            linecolor='rgba(0,0,0,0.2)',
            mirror=True,
            tickfont=dict(size=10, color='#2c3e50'),
            side='right'
        )
        
        # Specific axis configurations
        fig.update_yaxes(
            title_text="Price (USDT)",
            title_font=dict(size=12, color='#2c3e50'),
            tickformat='.6f' if df['close'].iloc[-1] < 1 else '.4f',
            row=1, col=1
        )
        
        fig.update_yaxes(
            title_text="Volume",
            title_font=dict(size=12, color='#2c3e50'),
            row=2, col=1
        )
        
        fig.update_yaxes(
            title_text="Stoch RSI (%)",
            title_font=dict(size=12, color='#2c3e50'),
            range=[0, 100],
            row=3, col=1
        )
        
        fig.update_xaxes(
            title_text="Time",
            title_font=dict(size=12, color='#2c3e50'),
            row=3, col=1
        )
        
        return fig
    
    def add_technical_overlays(self, fig, df: pd.DataFrame):
        """Add technical analysis overlays"""
        try:
            # Moving averages
            for ma, color, width in [
                ('sma_20', '#2196f3', 2),
                ('sma_50', '#ff9800', 2),
                ('ema_12', '#9c27b0', 1.5),
                ('ema_26', '#e91e63', 1.5)
            ]:
                if ma in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index, 
                            y=df[ma], 
                            mode='lines', 
                            name=ma.upper().replace('_', ' '),
                            line=dict(color=color, width=width),
                            opacity=0.8
                        ),
                        row=1, col=1
                    )
            
            # VWAP
            if 'vwap' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, 
                        y=df['vwap'], 
                        mode='lines', 
                        name='VWAP',
                        line=dict(color='#795548', width=2, dash='dot'),
                        opacity=0.9
                    ),
                    row=1, col=1
                )
            
            # Bollinger Bands
            if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
                fig.add_trace(
                    go.Scatter(
                        x=df.index, 
                        y=df['bb_upper'], 
                        mode='lines', 
                        name='BB Upper',
                        line=dict(color='rgba(96, 125, 139, 0.5)', width=1),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index, 
                        y=df['bb_lower'], 
                        mode='lines', 
                        name='Bollinger Bands',
                        line=dict(color='rgba(96, 125, 139, 0.5)', width=1),
                        fill='tonexty',
                        fillcolor='rgba(96, 125, 139, 0.1)'
                    ),
                    row=1, col=1
                )
            
            # Support and Resistance
            if 'support' in df.columns and 'resistance' in df.columns:
                latest_support = df['support'].dropna().iloc[-1] if not df['support'].dropna().empty else 0
                latest_resistance = df['resistance'].dropna().iloc[-1] if not df['resistance'].dropna().empty else 0
                
                if latest_support > 0:
                    fig.add_hline(
                        y=latest_support, 
                        line_dash="dash", 
                        line_color="#4caf50", 
                        line_width=2,
                        annotation_text="Support", 
                        annotation_position="bottom right",
                        row=1, col=1
                    )
                
                if latest_resistance > 0:
                    fig.add_hline(
                        y=latest_resistance, 
                        line_dash="dash", 
                        line_color="#f44336", 
                        line_width=2,
                        annotation_text="Resistance", 
                        annotation_position="top right",
                        row=1, col=1
                    )
                    
        except Exception as e:
            self.logger.error(f"Error adding technical overlays: {e}")
    
    def add_volume_levels(self, fig, volume_profile: Dict):
        """Add volume profile levels"""
        try:
            if not volume_profile:
                return
            
            # POC - Point of Control
            if volume_profile.get('poc', 0) > 0:
                fig.add_hline(
                    y=volume_profile['poc'], 
                    line_dash="solid", 
                    line_color="#ffc107", 
                    line_width=3,
                    annotation_text="POC", 
                    annotation_position="bottom right",
                    row=1, col=1
                )
            
            # Value Area High and Low
            if volume_profile.get('vah', 0) > 0 and volume_profile.get('val', 0) > 0:
                fig.add_hrect(
                    y0=volume_profile['val'], 
                    y1=volume_profile['vah'],
                    fillcolor="rgba(255,193,7,0.1)", 
                    layer="below", 
                    line_width=0,
                    row=1, col=1
                )
                
                fig.add_hline(
                    y=volume_profile['vah'], 
                    line_dash="dot", 
                    line_color="#ff9800", 
                    line_width=1,
                    annotation_text="VAH", 
                    annotation_position="bottom right",
                    row=1, col=1
                )
                
                fig.add_hline(
                    y=volume_profile['val'], 
                    line_dash="dot", 
                    line_color="#ff9800", 
                    line_width=1,
                    annotation_text="VAL", 
                    annotation_position="top right",
                    row=1, col=1
                )
                
        except Exception as e:
            self.logger.error(f"Error adding volume levels: {e}")
    
    def add_fibonacci_levels(self, fig, fibonacci_data: Dict):
        """Add Fibonacci retracement levels"""
        try:
            if not fibonacci_data or not fibonacci_data.get('levels'):
                return
            
            levels = fibonacci_data['levels']
            colors = {
                '23.6%': '#ff9800',
                '38.2%': '#2196f3', 
                '50.0%': '#f44336',
                '61.8%': '#4caf50',
                '78.6%': '#9c27b0'
            }
            
            for level_name, price in levels.items():
                if price > 0 and level_name in colors:
                    fig.add_hline(
                        y=price, 
                        line_dash="dashdot", 
                        line_color=colors[level_name], 
                        line_width=1,
                        opacity=0.7,
                        annotation_text=f"Fib {level_name}", 
                        annotation_position="bottom right",  # FIXED: was "middle right"
                        row=1, col=1
                    )
                    
        except Exception as e:
            self.logger.error(f"Error adding Fibonacci levels: {e}")
    
    def add_confluence_zones(self, fig, confluence_zones: List[Dict]):
        """Add confluence zone highlights"""
        try:
            if not confluence_zones:
                return
            
            # Add top 3 confluence zones
            for i, zone in enumerate(confluence_zones[:3]):
                if zone.get('price', 0) <= 0:
                    continue
                
                color = ['#4caf50', '#ff9800', '#f44336'][i]
                alpha = [0.3, 0.2, 0.1][i]
                
                fig.add_hline(
                    y=zone['price'], 
                    line_dash="solid", 
                    line_color=color, 
                    line_width=2,
                    opacity=0.8,
                    annotation_text=f"Confluence {i+1}", 
                    annotation_position="bottom left",
                    row=1, col=1
                )
                
        except Exception as e:
            self.logger.error(f"Error adding confluence zones: {e}")
    
    def add_signal_annotations(self, fig, signal_data: Dict, df: pd.DataFrame):
        """Add comprehensive signal annotations with TP/SL boxes"""
        try:
            if not signal_data:
                return
            
            entry_price = signal_data.get('entry_price', 0)
            stop_loss = signal_data.get('stop_loss', 0)
            take_profit_1 = signal_data.get('take_profit_1', 0)
            take_profit_2 = signal_data.get('take_profit_2', 0)
            side = signal_data.get('side', 'buy')
            confidence = signal_data.get('confidence', 0)
            
            # Get MTF data
            mtf_analysis = signal_data.get('mtf_analysis', {})
            confirmed_timeframes = mtf_analysis.get('confirmed_timeframes', [])
            mtf_status = signal_data.get('mtf_status', 'UNKNOWN')
            
            if entry_price <= 0:
                return
            
            # Signal marker
            marker_color = '#4caf50' if side.lower() == 'buy' else '#f44336'
            marker_symbol = 'triangle-up' if side.lower() == 'buy' else 'triangle-down'
            
            fig.add_trace(
                go.Scatter(
                    x=[df.index[-1]],
                    y=[entry_price],
                    mode='markers',
                    marker=dict(
                        symbol=marker_symbol,
                        size=20,
                        color=marker_color,
                        line=dict(width=3, color='white')
                    ),
                    name=f"{side.upper()} Signal",
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Entry line
            fig.add_hline(
                y=entry_price,
                line_dash="solid",
                line_color=marker_color,
                line_width=3,
                annotation_text=f"Entry: ${entry_price:.6f}",
                annotation_position="bottom left",
                row=1, col=1
            )
            
            # Stop Loss
            if stop_loss > 0:
                fig.add_hline(
                    y=stop_loss,
                    line_dash="dash",
                    line_color="#f44336",
                    line_width=2,
                    annotation_text=f"Stop Loss: ${stop_loss:.6f}",
                    annotation_position="bottom right",
                    row=1, col=1
                )
            
            # Take Profit levels
            if take_profit_1 > 0:
                fig.add_hline(
                    y=take_profit_1,
                    line_dash="dash",
                    line_color="#4caf50",
                    line_width=2,
                    annotation_text=f"TP1: ${take_profit_1:.6f}",
                    annotation_position="top right",
                    row=1, col=1
                )
            
            if take_profit_2 > 0:
                fig.add_hline(
                    y=take_profit_2,
                    line_dash="dash",
                    line_color="#2e7d32",
                    line_width=2,
                    annotation_text=f"TP2: ${take_profit_2:.6f}",
                    annotation_position="top left",
                    row=1, col=1
                )
            
            # Signal info box
            signal_info = (
                f"<b>{side.upper()} SIGNAL</b><br>"
                f"Confidence: {confidence:.1f}%<br>"
                f"MTF Status: {mtf_status}<br>"
                f"Confirmed: {'/'.join(confirmed_timeframes) if confirmed_timeframes else 'None'}<br>"
                f"Type: {signal_data.get('order_type', '').upper()}"
            )
            
            fig.add_annotation(
                text=signal_info,
                x=0.02, y=0.98,
                xref="x domain", yref="y domain",
                xanchor="left", yanchor="top",
                bgcolor=f"rgba{(*[int(marker_color[i:i+2], 16) for i in (1, 3, 5)], 0.9)}",
                bordercolor=marker_color,
                borderwidth=2,
                font=dict(size=12, color="white", family="Arial Black"),
                row=1, col=1
            )
            
        except Exception as e:
            self.logger.error(f"Error adding signal annotations: {e}")