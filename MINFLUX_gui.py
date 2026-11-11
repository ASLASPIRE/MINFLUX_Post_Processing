"""
3D Analysis Pipeline - CustomTkinter + Plotly boilerplate
=========================================================

SETUP
    pip install requirements.txt
    python MINFLUX_gui.py

HOW PLOTLY FITS INSIDE TKINTER 
  Plotly renders with JavaScript, and Tkinter has no web view. So:

    [ Tk window ]  <--- queue --->  [ tiny local web server ]  <--- HTTP --->  [ plot window ]
     (controls)                      (127.0.0.1, random port)                  (pywebview, or
                                                                                your browser)

  * Python builds the figure with plain px.scatter_3d(...) and PUBLISHES it to
    the server.
  * The plot window (a native pywebview window, launched as a subprocess so it
    doesn't fight Tk for the main loop) loads a small HTML page that draws the
    figure with plotly.js, and polls for updates.
  * When you click a point, the page POSTs its id to /delete. The server
    removes it from the data, republishes the figure, and the page redraws.
  * If pywebview isn't installed, the plot opens in your default browser
    instead. Everything else works identically.
"""

import sys

# ---------------------------------------------------------------------------
# Plot-window mode: `python pipeline_app.py --viewer URL`
# Handled first so the viewer process stays light (no Tk, no pandas).
# ---------------------------------------------------------------------------
def run_viewer(url: str):
    try:
        import webview
    except ImportError:
        import webbrowser, time
        webbrowser.open(url)
        while True:           # keep this process alive so the parent can manage it
            time.sleep(3600)
    webview.create_window("3D Viewer", url, width=950, height=720,
                          background_color="#1e1e2e")
    webview.start()


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "--viewer":
    run_viewer(sys.argv[2])
    sys.exit(0)

# ---------------------------------------------------------------------------
import json
import queue
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tkinter import filedialog, messagebox
import plotly
import customtkinter as ctk
import numpy as np
import plotly.express as px
from plotly.offline import get_plotlyjs
import math 
import os
import glob
import pandas as pd

import plotly
import plotly.io as pio
import plotly.figure_factory as ff
import plotly.graph_objects as go

import matplotlib.pyplot as plt

import numpy as np
import math
import statistics

from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
import sys
from descartes import PolygonPatch
import trimesh
sys.path.insert(0, os.path.dirname(os.getcwd()))
from sklearn import metrics
from sklearn.cluster import DBSCAN
import alphashape

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG, PANEL, FG, ACCENT = "#1e1e2e", "#313244", "#cdd6f4", "#89b4fa"
XLSX_NAME = "coordinates.xlsx"


# ===========================================================================
# MINFLUX ANALYSIS BEGINS HERE
# ===========================================================================
def run_analysis(df: pd.DataFrame, sigmas: tuple[float, float, float]):
    """
    PLACEHOLDER - replace the body with your real analysis.

    Input
      df      DataFrame with columns x, y, z (one row per point)
      sigmas  (sigma_x, sigma_y, sigma_z) as entered by the user

    Output (the contract the rest of the app relies on)
      result  DataFrame with AT LEAST columns x, y, z - these are the points
              shown in the second plot and written to coordinates.xlsx.
              Extra columns are kept in the Excel file.
      summary dict of {label: value} - printed in the results panel.
    """
    sx, sy, sz = sigmas

    mean = df[["x", "y", "z"]].mean()
    std = df[["x", "y", "z"]].std()

    # Example: keep points within (sigma * std) of the mean on every axis.
    keep = (
        ((df["x"] - mean["x"]).abs() <= sx * std["x"])
        & ((df["y"] - mean["y"]).abs() <= sy * std["y"])
        & ((df["z"] - mean["z"]).abs() <= sz * std["z"])
    )
    result = df[keep].copy()
    result["distance_from_mean"] = np.sqrt(
        ((result[["x", "y", "z"]] - mean) ** 2).sum(axis=1)
    )

    summary = {
        "Input points": len(df),
        "Points kept": len(result),
        "Points rejected": len(df) - len(result),
        "Mean (x, y, z)": tuple(round(v, 4) for v in mean),
        "Std (x, y, z)": tuple(round(v, 4) for v in std),
        "Sigma used (x, y, z)": sigmas,
    }
    return result, summary


# ===========================================================================
# FIGURES  (plain Plotly Express - edit styling freely)
# ===========================================================================
def make_scatter3d(df: pd.DataFrame, title: str, labels: dict, marker_size: int = 4, color_col: str | None = None):
    fig = px.scatter_3d(df, x="x", y="y", z="z", color=color_col,title=title, labels=labels)
    marker = dict(size=marker_size, line=dict(width=0))
    if color_col and color_col in df:
        marker.update(color=df[color_col].tolist(), colorscale="Viridis",
                      showscale=True, colorbar=dict(title=color_col, thickness=12))
    else:
        marker.update(color=ACCENT)
    # customdata carries each point's id so a click can say WHICH point it was.
    if "id" in df:
        fig.update_traces(marker=marker, customdata=df["id"].tolist())
    else:
        fig.update_traces(marker=marker) #when visualization the TID filter not necessary
    fig.update_layout(
        paper_bgcolor=BG,
        margin=dict(l=0, r=0, t=50, b=0),
        uirevision="keep",        # keep the camera angle when the data changes
        scene=dict(bgcolor=BG),
    )
    return fig

def make_alphashape(pointcloud, alpha: float = 0.001):
    import alphashape
    figALphaShape = plt.figure()
    axAlphaShape = plt.axes(projection = '3d')
    alpha_shape = alphashape.alphashape(pointcloud, alpha)
    axAlphaShape.plot_trisurf(*zip(*alpha_shape.vertices), triangles=alpha_shape.faces)
    xcoor_alpha, ycoor_alpha, zcoor_alpha = alpha_shape.vertices.T
    figAlphaShapeplot = ff.create_trisurf(
        x=xcoor_alpha,
        y=ycoor_alpha,
        z=zcoor_alpha,
        simplices = alpha_shape.faces, 
        plot_edges=False,
        edges_color='gray',
        colormap='Viridis')

    return figAlphaShapeplot, alpha_shape
# ===========================================================================
# THE LOCAL PLOT SERVER
# ===========================================================================
PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>3D Viewer</title>
<style>
 html,body{margin:0;height:100%;background:#1e1e2e;color:#cdd6f4;font-family:system-ui,sans-serif}
 #plot{width:100%;height:100%}
 #hint{position:absolute;top:12px;left:50%;transform:translateX(-50%);z-index:10;display:none;
       padding:6px 16px;border-radius:999px;background:#f38ba8;color:#1e1e2e;font-size:13px;font-weight:600}
 #wait{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;opacity:.6}
</style><script src="/plotly.js"></script></head><body>
<div id="hint">Click a point to delete it</div>
<div id="wait">Waiting for data…</div>
<div id="plot"></div>
<script>
let last = -1, started = false, editable = false;
const gd = document.getElementById('plot');

async function refresh() {
  const s = await (await fetch('/state')).json();
  last = s.version;
  if (!s.figure) return;
  document.getElementById('wait').style.display = 'none';
  editable = s.editable;
  document.getElementById('hint').style.display = editable ? 'block' : 'none';
  if (!started) {
    await Plotly.newPlot(gd, s.figure.data, s.figure.layout, {responsive: true});
    started = true;
    gd.on('plotly_click', onClick);
  } else {
    Plotly.react(gd, s.figure.data, s.figure.layout);
  }
}
let lastClick = 0;
async function onClick(ev) {
  if (!editable || !ev.points.length) return;
  // plotly_click can fire twice per click in 3D; the 2nd would pick a neighbour after the redraw
  const now = Date.now();
  if (now - lastClick < 500) return;
  lastClick = now;
  let id = ev.points[0].customdata;
  if (Array.isArray(id)) id = id[0];
  await fetch('/delete', {method: 'POST', headers: {'Content-Type': 'application/json'},
                          body: JSON.stringify({id})});
  lastClick = Date.now();
  refresh();
}
async function poll() {
  try {
    const v = parseInt(await (await fetch('/version')).text());
    if (v !== last) await refresh();
  } catch (e) {}
  setTimeout(poll, 600);
}
poll();
</script></body></html>"""


class PlotServer:
    """Serves the viewer page and the current figure; reports clicks back."""

    def __init__(self, on_delete):
        self._lock = threading.Lock()
        self._version = 0
        self._payload = '{"version":0,"editable":false,"figure":null}'
        self._plotly_js = get_plotlyjs().encode("utf-8")   # served locally: works offline
        self._on_delete = on_delete                        # called with a point id
        self.path_folder = ""
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):          # keep the console quiet
                pass

            def _send(self, body: bytes, ctype: str):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/":
                    self._send(PAGE.encode(), "text/html; charset=utf-8")
                elif path == "/plotly.js":
                    self._send(server._plotly_js, "application/javascript")
                elif path == "/version":
                    with server._lock:
                        self._send(str(server._version).encode(), "text/plain")
                elif path == "/state":
                    with server._lock:
                        self._send(server._payload.encode(), "application/json")
                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path != "/delete":
                    return self.send_error(404)
                length = int(self.headers.get("Content-Length", 0))
                try:
                    point_id = int(json.loads(self.rfile.read(length))["id"])
                    ok = server._on_delete(point_id)
                except Exception:
                    ok = False
                self._send(json.dumps({"ok": ok}).encode(), "application/json")

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)   # port 0 = pick a free one
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}/"
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def publish(self, fig, editable: bool):
        """Make `fig` the figure every connected viewer shows."""
        with self._lock:
            self._version += 1
            self._payload = (
                f'{{"version":{self._version},'
                f'"editable":{"true" if editable else "false"},'
                f'"figure":{fig.to_json()}}}'
            )

    def shutdown(self):
        self._httpd.shutdown()


# ===========================================================================
# THE APP
# ===========================================================================
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("MINFLUX Analysis Pipeline")
        self.geometry("1100x720")
        self.minsize(900, 600)

        # ---- app state (plain data, separate from widgets) ----
        self.csv_path: Path | None = None
        self.raw_df: pd.DataFrame | None = None        # columns: id, x, y, z
        self.labels = {"x": "x", "y": "y", "z": "z"}   # original CSV column names
        self.analysis_df: pd.DataFrame | None = None   # points in the 2nd plot
        self.nearest_neighbour_distances: list[float] | None = None
        self.undo_stack: list[pd.DataFrame] = []
        self.data_lock = threading.Lock()   # the server thread deletes points too
        self.events: queue.Queue[str] = queue.Queue()  # server thread -> Tk thread
        self.viewer: subprocess.Popen | None = None
        self.tid_filtered_data: pd.DataFrame | None = None
        self.dict_for_stats: dict | None = None #dictionary used to keep the variables that will be used for analysis later
        self.server = PlotServer(on_delete=self._on_point_deleted)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_steps_panel()
        self._build_results_panel()
        self._update_states()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(200, self._poll_events)

    # -----------------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------------
    def _section(self, parent, number: int, title: str) -> ctk.CTkFrame:
        """A rounded card with a numbered badge. Returns the body frame."""
        card = ctk.CTkFrame(parent, corner_radius=12)
        card.pack(fill="x", padx=12, pady=6)
        header = ctk.CTkFrame(card, fg_color="transparent")
        header.pack(fill="x", padx=14, pady=(12, 4))
        ctk.CTkLabel(header, text=str(number), width=26, height=26, corner_radius=13,
                     fg_color=ACCENT, text_color=BG,
                     font=ctk.CTkFont(weight="bold")).pack(side="left")
        ctk.CTkLabel(header, text=title,
                     font=ctk.CTkFont(size=15, weight="bold")).pack(side="left", padx=10)
        body = ctk.CTkFrame(card, fg_color="transparent")
        body.pack(fill="x", padx=14, pady=(0, 14))
        return body

    def _build_steps_panel(self):
        panel = ctk.CTkScrollableFrame(self, width=360, corner_radius=0, fg_color="transparent")
        panel.grid(row=0, column=0, sticky="nsw")

        ctk.CTkLabel(panel, text="3D Analysis Pipeline",
                     font=ctk.CTkFont(size=22, weight="bold")).pack(anchor="w", padx=16, pady=(16, 6))

        # --- 1. Upload CSV ---
        body = self._section(panel, 1, "Upload CSV")
        self.btn_upload = ctk.CTkButton(body, text="Choose file…", command=self.load_csv)
        self.btn_upload.pack(fill="x")
        self.lbl_file = ctk.CTkLabel(body, text="No file loaded", text_color="gray60",
                                     anchor="w", wraplength=300, justify="left")
        self.lbl_file.pack(fill="x", pady=(6, 0))

       

        # --- 2. Visualize Raw Data---
        body = self._section(panel, 2, "Visualize raw data")
        self.btn_visualize = ctk.CTkButton(body, text="Show raw 3D plot", command=self.visualize_raw)
        self.btn_visualize.pack(fill="x")

        # --- 3. Filter by TIDs and Visualize ---
        body = self._section(panel, 3, "Filter by TIDs and Visualize")
        self.btn_filter_visualize = ctk.CTkButton(body, text="Filter and show 3D plot", command=self.aggregate_localizations_from_TID)
        self.btn_filter_visualize.pack(fill="x")

        # --- 4 Sigma values ---
        body = self._section(panel, 4, "Sigma values")
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x")
        self.sigma_entries: list[ctk.CTkEntry] = []
        for i, name in enumerate(("σx", "σy", "σz")):
            row.grid_columnconfigure(i, weight=1)
            cell = ctk.CTkFrame(row, fg_color="transparent")
            cell.grid(row=0, column=i, padx=3, sticky="ew")
            ctk.CTkLabel(cell, text=name, text_color="gray70").pack(anchor="w")
            entry = ctk.CTkEntry(cell, placeholder_text="3.0")
            entry.insert(0, "3.0")
            entry.pack(fill="x")
            self.sigma_entries.append(entry)

        # --- 5. DBSCAN + Visualization with Click-to-Remove Outlier Functionality ---
        body = self._section(panel, 5, "Run DBSCANanalysis")
        self.btn_analyze = ctk.CTkButton(body, text="Run analysis + show result",
                                         command=self.run_dbscan)
        self.btn_analyze.pack(fill="x")
        ctk.CTkLabel(body, text="Then click points in the plot to delete them.",
                     text_color="gray60", anchor="w", wraplength=300,
                     justify="left").pack(fill="x", pady=(6, 0))

        # --- 6. Save ---
        body = self._section(panel, 6, "Save coordinates")
        self.btn_undo = ctk.CTkButton(body, text="Undo last delete", command=self.undo,
                                      fg_color="transparent", border_width=2)
        self.btn_undo.pack(fill="x", pady=(0, 6))
        self.btn_save = ctk.CTkButton(body, text=f"Save {XLSX_NAME}", command=self.save_xlsx)
        self.btn_save.pack(fill="x")

        # --- 7. Alphashape Visualization ---
        body = self._section(panel, 7, "Alphashape Visualization")
        self.btn_alphashape = ctk.CTkButton(body, text="Show Alphashape", command=self.visualize_alphashape)
        self.btn_alphashape.pack(fill="x")

        #-- 8. Nanocluster Analysis & Visualization --
        body = self._section(panel, 8, "Nanocluster Analysis & Visualization")
        self.btn_nanocluster = ctk.CTkButton(body, text="Show Nanoclusters", command=self.nanocluster_analysis)
        self.btn_nanocluster.pack(fill="x")

    def _build_results_panel(self):
        panel = ctk.CTkFrame(self, corner_radius=12)
        panel.grid(row=0, column=1, sticky="nsew", padx=(0, 16), pady=16)
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(panel, text="Results", font=ctk.CTkFont(size=16, weight="bold")
                     ).grid(row=0, column=0, sticky="w", padx=16, pady=(14, 6))
        self.log_box = ctk.CTkTextbox(panel, font=ctk.CTkFont(family="Courier", size=13),
                                      state="disabled", wrap="word")
        self.log_box.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 6))

        self.lbl_status = ctk.CTkLabel(panel, text="Ready.", text_color="gray70", anchor="w")
        self.lbl_status.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 12))

        self.log("Welcome! Start by uploading a CSV (step 1).")

    # -----------------------------------------------------------------------
    # Small helpers
    # -----------------------------------------------------------------------
    def log(self, text: str, clear: bool = False):
        self.log_box.configure(state="normal")
        if clear:
            self.log_box.delete("1.0", "end")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _read_sigmas(self) -> tuple[float, float, float]:
        try:
            values = tuple(float(e.get()) for e in self.sigma_entries)
        except ValueError:
            raise ValueError("Sigma values must be numbers.")
        if any(v <= 0 for v in values):
            raise ValueError("Sigma values must be greater than zero.")
        return values  # type: ignore[return-value]

    def _update_states(self):
        """Enable each button only when its prerequisites are met."""
        has_raw = self.raw_df is not None
        has_result = self.analysis_df is not None
        self.btn_visualize.configure(state="normal" if has_raw else "disabled")
        self.btn_filter_visualize.configure(state="normal" if has_raw else "disabled")
        self.btn_filter_visualize.configure(state="normal" if has_raw else "disabled")
        self.btn_analyze.configure(state="normal" if has_raw else "disabled")
        self.btn_save.configure(state="normal" if has_result else "disabled")
        self.btn_undo.configure(state="normal" if self.undo_stack else "disabled")

    def _open_viewer(self):
        """Open the plot window once; reopen if the user closed it."""
        if self.viewer is None or self.viewer.poll() is not None:
            self.viewer = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--viewer", self.server.url])

    def _poll_events(self):
        """Runs on the Tk thread every 200 ms; drains messages from the server thread."""
        try:
            while True:
                self.log(self.events.get_nowait())
        except queue.Empty:
            pass
        self._update_states()
        self.after(200, self._poll_events)

    # -----------------------------------------------------------------------
    # Pipeline steps
    # -----------------------------------------------------------------------
    def load_csv(self):  # Step 1
        #Initialize the df for stats
        self.dict_for_stats = {}
        path = filedialog.askopenfilename(
            title="Choose a CSV file", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            df = pd.read_csv(path)
            numeric = df.select_dtypes("number").dropna()
            if numeric.shape[1] < 3:
                raise ValueError("The CSV needs at least three numeric columns (x, y, z).")
            # numeric = numeric.iloc[:, :3]
            names = ['x', 'y', 'z']
            # raw = numeric.set_axis(["x", "y", "z"], axis=1).reset_index(drop=True)
            df.insert(0, "id", range(len(df)))
            raw = df
        except Exception as exc:
            messagebox.showerror("Could not load CSV", str(exc))
            return

        self.csv_path, self.raw_df = Path(path), raw
        self.labels = dict(zip("xyz", names))
        self.analysis_df, self.undo_stack = None, []
        self.lbl_file.configure(text=f"{self.csv_path.name}  ({len(raw)} points)")
        self.path_folder = str(self.csv_path.parent)
        self.log(f"Loaded {len(raw)} points; using columns {names} as x, y, z.", clear=True)
        self._update_states()

    def visualize_raw(self):  # Step 2
        fig = make_scatter3d(self.raw_df, "Raw data", self.labels,color_col='tid')
        plotly.offline.plot(fig, filename= self.path_folder + f'/{self.csv_path.stem}_rawfile.html',auto_open=False)
        self.server.publish(fig, editable=False)
        self._open_viewer()
        self.log("Showing raw 3D data in the plot window.")

    def aggregate_localizations_from_TID(self): #Step 3
        #Refactoring for this section to fit this GUI
        ### Rejecting all TIDs with localisations less than 5
        datafile = self.raw_df.copy()  # Use the raw_df from the GUI
        datafile_filter = datafile.groupby(datafile.tid.tolist(),as_index=False).size()
        mask = datafile_filter['size'] <=4
        datafile_filter2 = datafile_filter[~mask]
        self.log(f"Filtered TIDs with at least 5 localizations:\n{datafile_filter2}")
        if self.dict_for_stats is not None:
            self.dict_for_stats["filtered_TID_count"] = len(datafile_filter2)
        self.log(f"self.dict_for_stats:\n{self.dict_for_stats}")    
        ##The number of rows in datafile_filter2 will be filtered_TID_count in our stats df
        if self.dict_for_stats is not None:
            self.dict_for_stats["filtered_TID_count"] = len(datafile_filter2)
        self.log(f"self.dict_for_stats:\n{self.dict_for_stats}")
        ### Aggregating localisations based on TID (average of coordinates/TID gives centroid/TID)
        Tidgroupcoord = datafile.groupby('tid', as_index=False)
        datafile_TIDsorted = Tidgroupcoord.agg({'x':'mean', 'y':'mean', 'z':'mean'})
        self.log(f"Aggregated localizations by TID:\n{datafile_TIDsorted}")

        ## Applying filter of TIDs less than 5 
        TIDmask = datafile_TIDsorted['tid'].isin(datafile_filter2['index'])
        datafile_TIDsorted_filtered = datafile_TIDsorted[TIDmask]
        self.log(f"Filtered aggregated localizations by TID:\n{datafile_TIDsorted_filtered}")
        data = datafile_TIDsorted_filtered.drop(['tid'], axis=1)
        self.tid_filtered_data = data
        fig = make_scatter3d(data, "Filtered Aggregated Localizations by TID", self.labels, color_col='z')
        self.server.publish(fig, editable=False)
        self._open_viewer()
        self.log("Showing filtered aggregated 3D data in the plot window.")
        self.log("Saving 3D scatter plot of filtered aggregated localizations by TID.")
        fig.write_image("rawfile_TIDsorted.png")
        plotly.offline.plot(fig, filename= self.path_folder + f'/{self.csv_path.stem}_rawfile_TIDsorted.html',auto_open=False)
        self.log(f"3D scatter plot HTML saved to {self.path_folder + f'/{self.csv_path.stem}_rawfile_TIDsorted.html'}")
        self.log("3D scatter plot saved.")
    
    def run_dbscan(self):  
        try:
            sigmas = self._read_sigmas() #Step 4
        except ValueError as exc:
            messagebox.showerror("Invalid sigma values", str(exc))
            return
        try:
            result, summary = run_analysis(self.raw_df[["x", "y", "z"]].copy(), sigmas)
        except Exception as exc:
            messagebox.showerror("Analysis failed", f"{type(exc).__name__}: {exc}")
            return
        epsilon = math.sqrt(2*sigmas[0] +2* sigmas[1] + 2*sigmas[2])
        #Step 5: Run DBSCAN
        self.log(f"Running DBSCAN with epsilon={epsilon}")
        db_model = DBSCAN(eps=epsilon, min_samples=1).fit(self.tid_filtered_data)  
        self.log("DBSCAN clustering completed.")
        labels = db_model.labels_
        self.tid_filtered_data['cluster number']=db_model.labels_

        ## print estimated number of clusters and number of noise points
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        self.log("Estimated number of clusters: %d" % n_clusters_)
        self.log("Estimated number of noise points: %d" % n_noise_)
        #Add the number of clusters and noise points to the statistics DF
        if self.dict_for_stats is not None:
            self.dict_for_stats["n_clusters"] = n_clusters_
            self.dict_for_stats["n_noise"] = n_noise_
            self.log(f"Updated dict_for_stats with n_clusters and n_noise:\n{self.dict_for_stats}")
        
        #Plotting DBSCAN results (molecular counts)
        for x in range(0,len(set(self.tid_filtered_data['cluster number']))):
            globals()['cluster%s'%x] = self.tid_filtered_data[self.tid_filtered_data['cluster number']==x]
        
        #outlier = self.tid_filtered_data[self.tid_filtered_data['cluster number']==-1] ##This would only be relevant if min_samples > 1 for DBSCAN 

        ######### for each cluster calculate the coordinates of the center of mass
        ### center of mass coordinates are stored in lists called centroid(x)
        ### centroid lists are compiled in a numpy array, further arranged in a dataframe
        ### dataframe is plotted in 3D scatter plot and also as plotly express scatter plot
            
            
        for j in range(0,len(set(self.tid_filtered_data['cluster number']))): 
            globals()['cx%s' %j] = sum(globals()['cluster%s' %j].x)/len(globals()['cluster%s' %j].x)
            globals()['cy%s' %j] = sum(globals()['cluster%s' %j].y)/len(globals()['cluster%s' %j].y)
            globals()['cz%s' %j] = sum(globals()['cluster%s' %j].z)/len(globals()['cluster%s' %j].z)
            globals()['centroid%s' %j] = [globals()['cx%s' %j], globals()['cy%s' %j], globals()['cz%s' %j]]
                    ### this step gives individual centroids of each localisation cluster
                
        centroidList =  np.zeros(shape=((len(set(self.tid_filtered_data['cluster number']))),3))
        for j in range(0,len(set(self.tid_filtered_data['cluster number']))):
            centroidList[j,]=globals()['centroid%s' %j]

        centroidListDF = pd.DataFrame(centroidList)
        centroidListDF['id'] = range(len(centroidListDF)) #to allow for click deletion of points
        centroidListDF.rename(columns={0 : 'x', 1 : 'y', 2 :'z'}, inplace=True)
        #resetting the analysis dataframe to the centroid list dataframe
        self.analysis_df = centroidListDF.copy()
        self.undo_stack = []

        self._publish_analysis()
        self._open_viewer()
        self.log("Showing DBSCAN Centroid Scatter Plot")
        
        result = centroidListDF.copy().reset_index(drop=True)
        result["id"] = range(len(result))          # ids power click-to-delete
        with self.data_lock:
            self.analysis_df = result
            self.undo_stack = []
            self._publish_analysis()

        self.log("── Analysis results ──────────────", clear=True)
        for key, value in summary.items():
            self.log(f"{key:<24} {value}")
        self.log("\nClick a point in the plot window to delete it.")
        self._open_viewer()
        self._update_states()

    def visualize_alphashape(self):
    
        #calculating shortest neighbor distance
        centroidList = self.analysis_df[['x', 'y', 'z']].copy().to_numpy()
        Alldistances = distance.pdist(centroidList, 'euclidean')
        NearestAll = NearestNeighbors(n_neighbors=2, radius=max(Alldistances))
        NearestAll.fit(centroidList)
        distances, indices = NearestAll.kneighbors(centroidList)
        print("nearest neighbor distances =")
        #Add nearest neighbor distances to the statistics DF
       
        print (distances[:,1])
        NearestNeighbourDistances = distances[:,1]
        if self.dict_for_stats is not None:
            self.dict_for_stats["nearest_neighbour_distances"] = NearestNeighbourDistances
            self.log(f"Updated dict_for_stats with nearest_neighbour_distances:\n{self.dict_for_stats}")
        self.nearest_neighbour_distances = NearestNeighbourDistances
       #run alphashape creation
        alphashapefig, alpha_shape = make_alphashape(centroidList, 0.001)
        plotly.offline.plot(alphashapefig, filename= self.path_folder + f'/{self.csv_path.stem}_AlphaShape.html', auto_open=False)

        volume = alpha_shape.volume/1e+9
        if self.dict_for_stats is not None:
            self.dict_for_stats["alphashape_volume (µm³)"] = volume
            self.log(f"Updated dict_for_stats with alphashape_volume:\n{self.dict_for_stats}")
        #Density will be the number of counts per unit volume
        density = self.dict_for_stats["n_clusters"] / volume
        if self.dict_for_stats is not None:
            self.dict_for_stats["alphashape_density"] = density
            self.log(f"Updated dict_for_stats with alphashape_density:\n{self.dict_for_stats}")
        self.server.publish(alphashapefig, editable=False)
        self._open_viewer()

    def nanocluster_analysis(self):
        nanoclusterdistance = statistics.mean(self.nearest_neighbour_distances) + 2*statistics.stdev(self.nearest_neighbour_distances) 
        print(nanoclusterdistance)
        centroidListDF = self.analysis_df[['x', 'y', 'z']].copy()   # id must not be a clustering feature
        db_modelNano = DBSCAN(eps=nanoclusterdistance, min_samples=3).fit(centroidListDF)           ##### adjust here number of minimal points and epsilon values
        print(db_modelNano)
        labels_Nano = db_modelNano.labels_
        dataNanoCluster = centroidListDF.copy()
        dataNanoCluster['clusternumber']=db_modelNano.labels_

        ## print estimated number of clusters and number of noise points
        n_clusters_ = len(set(labels_Nano)) - (1 if -1 in labels_Nano else 0)
        n_noise_ = list(labels_Nano).count(-1)
        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)

        singlechannels = dataNanoCluster[dataNanoCluster['clusternumber']==-1]

        xcoordNano=dataNanoCluster.x
        ycoordNano=dataNanoCluster.y
        zcoordNano=dataNanoCluster.z
        NanoClusterNum=dataNanoCluster.clusternumber

        NanoClusterdistance = statistics.mean(self.nearest_neighbour_distances) + 2*statistics.stdev(self.nearest_neighbour_distances)

        db_Nanomodel = DBSCAN(eps=NanoClusterdistance, min_samples=3).fit(centroidListDF)           ##### adjust here number of minimal points and epsilon values
        print(db_Nanomodel)
        Nanolabels = db_Nanomodel.labels_
        dataNanoCluster = centroidListDF.copy()
        dataNanoCluster['clusternumber']=db_Nanomodel.labels_

        ## print estimated number of clusters and number of noise points
        n_clusters_ = len(set(Nanolabels)) - (1 if -1 in Nanolabels else 0)
        n_noise_ = list(Nanolabels).count(-1)
        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)

        clusteredchannels = dataNanoCluster[dataNanoCluster['clusternumber']!=-1]
        singlechannels = dataNanoCluster[dataNanoCluster['clusternumber']==-1]

        Nanoxcoord=clusteredchannels.x
        Nanoycoord=clusteredchannels.y
        Nanozcoord=clusteredchannels.z
        NanoClusterNum=clusteredchannels.clusternumber
        Singlexcoord=singlechannels.x
        Singleycoord=singlechannels.y
        Singlezcoord=singlechannels.z
        SinglechanNum=singlechannels.clusternumber

        fig2Nano = go.Figure(data=[go.Scatter3d
                                            (x=Nanoxcoord, y=Nanoycoord, z=Nanozcoord, 
                                                mode='markers',
                                                marker = dict
                                                (size=10, 
                                                color=NanoClusterNum,
                                                colorscale='Inferno',
                                                opacity = 1,
                                                colorbar=dict(thickness=20, len=0.2)),
                                                )], 
                        layout = go.Layout(font=dict(size=18)))

        fig2Nano.add_trace(go.Scatter3d
                                (x=Singlexcoord, y=Singleycoord, z=Singlezcoord, ###here the points can be changed to for instance remove outliers if they don't have to be plotted
                                                mode='markers', 
                                                marker = dict
                                                (size=10, 
                                                color=SinglechanNum, 
                                                colorscale=[[0, 'grey'], [0.5, 'grey'], [1.0, 'grey']],
                                                opacity =1,
                                                showscale=True
                                                )
                                                ))

        fig2Nano.update_layout(
                                scene = dict(
                                xaxis_title = 'x',
                                yaxis_title = 'y',
                                zaxis_title = 'z',
                                xaxis = dict(nticks=10, showticklabels=True, gridcolor='grey', backgroundcolor="white"),
                                            yaxis = dict(nticks=10, showticklabels=True, gridcolor='grey', backgroundcolor="white"),
                                            zaxis = dict(nticks=5, showticklabels=True, gridcolor='grey', backgroundcolor="white"),
                                            aspectmode="cube"),
                                scene_camera_eye=dict(x=1.5, y=1.5, z=1.5),
                            width=500,
                            height=500,
                            font = dict(size=18))
        # self.server.publish(fig2Nano, editable=False)
        # self._open_viewer()
        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of single channels: %d" % n_noise_)
        plotly.offline.plot(fig2Nano, filename= f"{self.path_folder}/{self.csv_path.stem}_NanoPlot.html",auto_open=False)
        self.log(f"NanoPlot HTML saved to {f'{self.path_folder}/{self.csv_path.stem}_NanoPlot.html'}")
        ## Determining centroids of each cluster
        NanoClusterCentroidCoord = clusteredchannels.groupby('clusternumber', as_index=False).mean()
        NanoClusterCentroidCoordDF = NanoClusterCentroidCoord.drop(['clusternumber'], axis=1)
        NanoClusterCentroidList = NanoClusterCentroidCoordDF.values.tolist()

        if len(NanoClusterCentroidList) < 2:
            self.log(f"Need at least 2 nanoclusters, found {len(NanoClusterCentroidList)}. Try a larger eps or fewer min_samples.")
            return
        InterClusterDistAll = distance.pdist(NanoClusterCentroidList, 'euclidean')
        InterClusterDist = NearestNeighbors(n_neighbors=2, radius=max(InterClusterDistAll))
        InterClusterDist.fit(NanoClusterCentroidList)
        distances, indices = InterClusterDist.kneighbors(NanoClusterCentroidList)
        print("nearest neighbor distances between centroids of nanocluster =")
        print (distances[:,1])
        NearestNeighbourDistances = distances[:,1]

        # Determining number of channels in each nanocluster 

        groupNanoClusterID = clusteredchannels.groupby(clusteredchannels.clusternumber.tolist(), as_index=False).size()
        print(groupNanoClusterID)

        # Determining intra-cluster nearest neighbour distance 
        for x in range(0,len(set(clusteredchannels['clusternumber']))):
            globals()['NanoCluster%s'%x] = clusteredchannels[clusteredchannels['clusternumber']==x]
            globals()['NanoCluster%s'%x] = globals()['NanoCluster%s'%x].drop(['clusternumber'], axis=1)
            globals()['NanoClusterList%s'%x] = globals()['NanoCluster%s'%x].values.tolist()
            globals()['IntraClusterDistAll%s'%x] = distance.pdist(globals()['NanoClusterList%s'%x], 'euclidean')
            globals()['IntraClusterDist%s'%x] = NearestNeighbors(n_neighbors=2, radius=max(globals()['IntraClusterDistAll%s'%x]))
            globals()['IntraClusterDist%s'%x].fit(globals()['NanoClusterList%s'%x])
            globals()['distances%s'%x], globals()['indices%s'%x] = globals()['IntraClusterDist%s'%x].kneighbors(globals()['NanoClusterList%s'%x]) 
            

        IntraClusterDistanceList = list()
            
        
        for x in range(0,len(set(clusteredchannels['clusternumber']))):
            IntraClusterDistanceList.append(globals()['distances%s'%x][:,1])
            print(IntraClusterDistanceList)
            IntraClusterDistanceSummaryInv = pd.DataFrame(IntraClusterDistanceList)
            IntraClusterDistanceSummary = IntraClusterDistanceSummaryInv.transpose(copy=False)
            print(IntraClusterDistanceSummary)


        # Fitting a spherical fit on the nanoclusters by determining the maximum extent of the nanocluster


        # Calculating the centroid of nanocluster
        for j in range(0,len(set(clusteredchannels['clusternumber']))): 
            globals()['cxnano%s' %j] = sum(globals()['NanoCluster%s' %j].x)/len(globals()['NanoCluster%s' %j].x)
            globals()['cynano%s' %j] = sum(globals()['NanoCluster%s' %j].y)/len(globals()['NanoCluster%s' %j].y)
            globals()['cznano%s' %j] = sum(globals()['NanoCluster%s' %j].z)/len(globals()['NanoCluster%s' %j].z)
            globals()['centroidnano%s' %j] = [globals()['cxnano%s' %j], globals()['cynano%s' %j], globals()['cznano%s' %j]]
            globals()['centroidnano_df%s'%j]=pd.DataFrame(globals()['centroidnano%s'%j])
            globals()['centroidnano_df%s'%j]=globals()['centroidnano_df%s'%j].transpose(copy=False)
                        ### this step gives individual centroids of each localisation cluster


        # dropping index of NanoCluster dataframes
        for j in range(0,len(set(clusteredchannels['clusternumber']))): 
            globals()['NanoCluster%s' %j] = globals()['NanoCluster%s' %j].reset_index(drop=True)

        # determining radius of nanocluster, i.e. distance from all points of nanocluster and the centroid of nanocluster is calculated
        # the maximum distance will be taken as radius of the nanocluster spherical fit 
                                                        

        for j in range(0,len(set(clusteredchannels['clusternumber']))):
                for i in range(0, len(globals()['NanoCluster%s'%j])): 
                    globals()['dist_nano'+ str(j) + str(i)] = math.sqrt(
                                (  (globals()['centroidnano%s'%j][0])  -  ((globals()['NanoCluster%s'%j]).x[i])  ) **2
                                +(  (globals()['centroidnano%s'%j][1])  -  ((globals()['NanoCluster%s'%j]).y[i])  ) **2
                                +(  (globals()['centroidnano%s'%j][2])  -  ((globals()['NanoCluster%s'%j]).z[i])  ) **2
        )
                                                        

        for j in range(0,len(set(clusteredchannels['clusternumber']))):
                for i in range(0, len(globals()['NanoCluster%s'%j])):
                        globals()['nanodistanceList%s' %j] =  np.zeros(shape=((len(globals()['NanoCluster%s'%j]),)))


        for j in range(0,len(set(clusteredchannels['clusternumber']))):
                for i in range(0, len(globals()['NanoCluster%s'%j])):
                    globals()['nanodistanceList%s' %j][i,] = globals()['dist_nano'+ str(j) + str(i)]
        for j in range(0,len(set(clusteredchannels['clusternumber']))):
                    print(max(globals()['nanodistanceList%s' %j]))

        #Diameter of nanoclusters
        NanoClusterDiameterList = np.zeros(shape=len(set(clusteredchannels['clusternumber']),))
        NanoClusterRadiusList = np.zeros(shape=len(set(clusteredchannels['clusternumber']),))
        for x in range(0,len(set(clusteredchannels['clusternumber']))):
            NanoClusterDiameterList[x,] = 2*(max(globals()['nanodistanceList%s' %x]))
            NanoClusterRadiusList[x,]= (max(globals()['nanodistanceList%s' %x]))
        NanoClusterDiameterList_df = pd.DataFrame(NanoClusterDiameterList)
        NanoClusterRadiusList_df = pd.DataFrame(NanoClusterRadiusList)
        NanoClusterRadiusList_df.rename(columns={0:'r'}, inplace=True)
        NanoClusterCentroidList =  np.zeros(shape=((len(set(clusteredchannels['clusternumber']))),3))
        for j in range(0,len(set(clusteredchannels['clusternumber']))):
            NanoClusterCentroidList[j,]=globals()['centroidnano%s' %j]
        NanoClusterCentroidList_df = pd.DataFrame(NanoClusterCentroidList)


        concatframes = [NanoClusterCentroidList_df, NanoClusterRadiusList_df]
        xyzr_sphere_coordinates = pd.concat(concatframes, axis=1)
        xyzr_sphere_coordinates.rename(columns={0 : 'x', 1 : 'y', 2 :'z'}, inplace=True)

        #Fit a sphere of given radius and centroid 

        def ms(x, y, z, radius, resolution=50):
            """Return the coordinates for plotting a sphere centered at (x,y,z)"""
            u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]
            X = radius * np.cos(u)*np.sin(v) + x
            Y = radius * np.sin(u)*np.sin(v) + y
            Z = radius * np.cos(v) + z
            return (X, Y, Z)

        df_nano_fit = xyzr_sphere_coordinates

        data_nano_fit = []
        for index, row in df_nano_fit.iterrows():
            (x_pns_surface, y_pns_surface, z_pns_suraface) = ms(row.x, row.y, row.z, row.r)
            data_nano_fit.append(go.Surface(x=x_pns_surface, y=y_pns_surface, z=z_pns_suraface, opacity=0.2, colorscale='Viridis'))


        fig3Nano = go.Figure(data=data_nano_fit)


        fig3Nano.add_trace(go.Scatter3d
                        (x=Nanoxcoord, y=Nanoycoord, z=Nanozcoord, 
                                                mode='markers',
                                                marker = dict
                                                (size=10, 
                                                color=NanoClusterNum,
                                                colorscale='Inferno',
                                                opacity = 1,
                                                colorbar=dict(thickness=20, len=0.2)),
                                                ))

        fig3Nano.add_trace(go.Scatter3d
                                (x=Singlexcoord, y=Singleycoord, z=Singlezcoord, ###here the points can be changed to for instance remove outliers if they don't have to be plotted
                                                mode='markers', 
                                                marker = dict
                                                (size=10, 
                                                color=SinglechanNum, 
                                                colorscale=[[0, 'grey'], [0.5, 'grey'], [1.0, 'grey']],
                                                opacity =1,
                                                showscale=True
                                                )
                                                ))

        fig3Nano.update_layout(
                                scene = dict(
                                xaxis_title = 'x',
                                yaxis_title = 'y',
                                zaxis_title = 'z',
                                xaxis = dict(nticks=10, showticklabels=True, gridcolor='grey', backgroundcolor="white"),
                                            yaxis = dict(nticks=10, showticklabels=True, gridcolor='grey', backgroundcolor="white"),
                                            zaxis = dict(nticks=5, showticklabels=True, gridcolor='grey', backgroundcolor="white"),
                                            aspectmode="auto"),
                                scene_camera_eye=dict(x=1.5, y=1.5, z=1.5),
                            width=1700,
                            height=1200,
                            font = dict(size=18))
        plotly.offline.plot(fig3Nano, filename= f"{self.path_folder +'/'+ self.csv_path.stem}_NanoPlot_sph_final.html",auto_open=False)
        self.log(f"NanoPlot sph HTML saved to {self.path_folder +'/'+ self.csv_path.stem + '_NanoPlot_sph_final.html'}")

        fig3Nano.write_image("NanoPlot_sph.svg")
        fig3Nano.write_image("NanoPlot_sph.png")
        plotly.offline.plot(fig3Nano, filename= f"{self.path_folder +'/'+ self.csv_path.stem}_NanoPlot_sph_final.html",auto_open=False)
        self.log(f"NanoPlot sph HTML saved to {self.path_folder +'/'+ self.csv_path.stem + '_NanoPlot_sph_final.html'}")

        #Final plot 
        self.server.publish(fig3Nano, editable=False)
        stats_path = f"{self.csv_path.parent}/stats_{self.csv_path.stem}.xlsx"
        pd.DataFrame.from_dict(self.dict_for_stats, orient='index').to_excel(stats_path,sheet_name='Sheet1')
        #Write the list self.nearest_neighbour_distances to the Excel file such that each distance is recorded in a separate row under the sheet 'NearestNeighborDistances'
        if self.nearest_neighbour_distances is not None:
            with pd.ExcelWriter(stats_path, mode='a', engine='openpyxl') as writer:
                pd.DataFrame({'nearest_neighbour_distances': self.nearest_neighbour_distances}).to_excel(writer, sheet_name='NearestNeighborDistances', index=False)
        self.log(f"Saved statistics to {stats_path}")
        self._open_viewer()


    def _publish_analysis(self):
        """Rebuild + publish the analysis plot. Caller must hold data_lock."""
        fig = make_scatter3d(
            self.analysis_df,title="DBSCAN Centroid Scatter Plot",labels=['x','y','z'],color_col='z', marker_size=20)
        plotly.offline.plot(fig, filename= self.path_folder + f'/{self.csv_path.stem}_DBSCANresult.html',auto_open=False)
        self.log(f"DBSCAN result HTML saved to {self.path_folder + f'/{self.csv_path.stem}_DBSCANresult.html'}")

        self.server.publish(fig, editable=True)

    def _on_point_deleted(self, point_id: int) -> bool:
        """Called on the SERVER thread when the plot reports a click."""
        with self.data_lock:
            if self.analysis_df is None:
                return False
            mask = self.analysis_df["id"] == point_id
            if not mask.any():
                return False
            self.undo_stack.append(self.analysis_df[mask])
            self.analysis_df = self.analysis_df[~mask]
            remaining = len(self.analysis_df)
            self._publish_analysis()
        self.events.put(f"Deleted point #{point_id}. {remaining} points remain.")
        return True

    def undo(self):
        with self.data_lock:
            if not self.undo_stack:
                return
            restored = self.undo_stack.pop()
            self.analysis_df = pd.concat([self.analysis_df, restored]).sort_values("id")
            self._publish_analysis()
        self.log(f"Restored point #{int(restored['id'].iloc[0])}.")
        self._update_states()

    def save_xlsx(self):  # Step 5
        with self.data_lock:
            out = self.analysis_df.drop(columns="id").rename(columns=self.labels)
        path = self.csv_path.with_name(XLSX_NAME)
        try:
            out.to_excel(path, index=False)
        except Exception as exc:
            messagebox.showerror("Could not save", str(exc))
            return
        self.log(f"Saved {len(out)} points to {path}")
        self.lbl_status.configure(text=f"Saved {path.name}")

    def _on_close(self):
        if self.viewer and self.viewer.poll() is None:
            self.viewer.terminate()
        self.server.shutdown()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()





   