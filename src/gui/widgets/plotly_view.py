"""
Plotly view helper for embedding interactive charts in PyQt5.
"""

import json
from typing import Optional, Dict, Any

from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, QTimer

from plotly.offline import get_plotlyjs
from plotly.utils import PlotlyJSONEncoder


class PlotlyView(QWebEngineView):
    """QWebEngineView wrapper that renders Plotly figures via Plotly.react."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loaded = False
        self._pending_payload: Optional[str] = None

        # Connect signals BEFORE loading HTML
        self.loadStarted.connect(self._on_load_started)
        self.loadFinished.connect(self._on_load_finished)

        # Set size policy to expand and fill parent
        from PyQt5.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 300)


        self._init_html()

    def resizeEvent(self, event):
        """Handle resize events and trigger Plotly relayout."""
        super().resizeEvent(event)

        # Trigger Plotly to relayout after resize
        if self._loaded and self.page():
            resize_script = """
            if (typeof Plotly !== 'undefined') {
                Plotly.Plots.resize('plot');
            }
            """
            self.page().runJavaScript(resize_script)

    def _init_html(self):
        # Simple HTML with empty plot container
        html = """
        <!DOCTYPE html>
        <html>
          <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
              * {
                box-sizing: border-box;
              }
              html, body {
                width: 100%;
                height: 100%;
                margin: 0;
                padding: 0;
                overflow: hidden;
              }
              #plot {
                width: 100%;
                height: 100%;
                background-color: #f5f5f5;
                font-family: Arial, sans-serif;
                font-size: 16px;
                color: #666;
              }
            </style>
          </head>
          <body>
            <div id="plot">Loading Plotly...</div>
            <script>
              console.log("HTML loaded, waiting for Plotly.js...");
              // We'll load Plotly.js and set up updatePlot after this HTML loads
              window.plotReady = false;
            </script>
          </body>
        </html>
        """

        # Use setHtml with explicit base URL to avoid security issues
        from PyQt5.QtCore import QUrl
        self.setHtml(html, QUrl("about:blank"))

    def _on_load_started(self):
        pass

    def _on_load_finished(self, ok: bool):
        if not ok:
            print("ERROR: PlotlyView HTML failed to load")
            return

        # Load Plotly.js dynamically after HTML is loaded
        load_plotly_script = """
        (function() {
            var script = document.createElement('script');
            script.src = 'https://cdn.plot.ly/plotly-2.27.0.min.js';
            script.onload = function() {
                window.plotReady = true;
                window.updatePlot = function(data, layout, config) {
                    // Force layout to use full width and height
                    layout.width = undefined;
                    layout.height = layout.height || undefined;
                    layout.autosize = true;

                    var plotConfig = config || {
                        responsive: true,
                        displaylogo: false,
                        displayModeBar: true
                    };

                    Plotly.react('plot', data, layout, plotConfig).then(function() {
                        // Force Plotly to recalculate size after render
                        setTimeout(function() {
                            Plotly.Plots.resize('plot');
                        }, 100);
                    });
                };
                // Signal that we're ready
                document.getElementById('plot').innerHTML = '<div style="text-align:center;padding:20px;">Ready for plots</div>';
            };
            script.onerror = function() {
                document.getElementById('plot').innerHTML = '<div style="text-align:center;padding:20px;color:red;">Failed to load Plotly.js. Check internet connection.</div>';
            };
            document.head.appendChild(script);
        })();
        """

        self.page().runJavaScript(load_plotly_script, self._on_plotly_load_script_executed)

    def _on_plotly_load_script_executed(self, result):
        # Wait for Plotly.js to actually load from CDN
        QTimer.singleShot(2000, self._check_plotly_ready)

    def _check_plotly_ready(self):
        check_script = "window.plotReady === true"
        self.page().runJavaScript(check_script, self._on_plotly_ready_checked)

    def _on_plotly_ready_checked(self, ready):
        if ready:
            self._loaded = True
            if self._pending_payload:
                self._run_payload(self._pending_payload)
                self._pending_payload = None
        else:
            # Not ready yet, check again in 2 seconds
            QTimer.singleShot(2000, self._check_plotly_ready)

    def update_plot(self, data: Any, layout: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        payload = json.dumps(
            {"data": data, "layout": layout, "config": config},
            cls=PlotlyJSONEncoder
        )
        if not self._loaded:
            self._pending_payload = payload
            return
        self._run_payload(payload)

    def _run_payload(self, payload: str):
        if self.page() is None:
            print("ERROR: PlotlyView page is None")
            return
        script = f"var payload = {payload}; updatePlot(payload.data, payload.layout, payload.config);"
        self.page().runJavaScript(script)

    def _on_script_result(self, result):
        """Callback for JavaScript execution (for debugging)."""
        pass
