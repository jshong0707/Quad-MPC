import pyqtgraph as pg
from collections import deque

class Plotter:
    def __init__(
        self,
        data,
        titles,
        Big_title=None,
        n=1,
        interval=0.01,
        maxlen=400,
        ylims=None,
        num_curves=1,
        pens=None,
    ):
        """
        Real‐time plotter using PyQtGraph.

        Parameters:
        - data: mujoco.MjData (for timestamping)
        - titles: list of subplot titles, length = n
        - Big_title: window title
        - n: number of subplots
        - interval: min sim‐time between redraws
        - maxlen: max history length per curve
        - ylims: list of (ymin, ymax) tuples, length = n
        - num_curves: int or list giving how many curves per subplot
        - pens: list of color strings
        """
        self.data = data
        self.n = n
        self.interval = interval
        self.maxlen = maxlen

        # Normalize ylims to list of pairs
        if ylims is None:
            # auto‐scale if not provided
            self.ylims = [None] * n
        else:
            assert len(ylims) == n, "ylims must be length n"
            self.ylims = ylims

        # Normalize num_curves
        if isinstance(num_curves, int):
            self.num_curves = [num_curves] * n
        else:
            assert len(num_curves) == n
            self.num_curves = num_curves

        # Default pens
        default_pens = ['y','r','g','c','m','w']
        self.pens = pens or default_pens

        # Create window
        self.win = pg.GraphicsLayoutWidget(title=Big_title)
        self.win.resize(650, 1050)

        # Create subplots and curves
        self.plots = []
        self.curves = []
        for i in range(n):
            p = self.win.addPlot(row=i, col=0)
            p.setTitle(titles[i])
            p.showGrid(x=True, y=True)
            self.plots.append(p)
            cvs = []
            for j in range(self.num_curves[i]):
                pen = self.pens[j % len(self.pens)]
                cvs.append(p.plot(pen=pen))
            self.curves.append(cvs)

        # Buffers
        self.ts = deque(maxlen=maxlen)
        # ys[i][j] is history deque for j-th curve on subplot i
        self.ys = [[deque(maxlen=maxlen) for _ in range(self.num_curves[i])]
                   for i in range(n)]

        self._last_update = 0.0

    def push(self, vals):
        """
        Append new values. For subplot i:
        - if num_curves[i]==1, vals[i] is scalar
        - else vals[i] is iterable of length num_curves[i]
        """
        t = self.data.time
        self.ts.append(t)
        for i in range(self.n):
            vi = vals[i]
            if self.num_curves[i] == 1:
                self.ys[i][0].append(float(vi))
            else:
                assert len(vi) == self.num_curves[i]
                for j, v in enumerate(vi):
                    self.ys[i][j].append(float(v))

    def update(self):
        """
        Redraw if sim‐time interval has passed.
        """
        if not self.ts:
            return
        now = self.data.time
        if now - self._last_update < self.interval:
            return
        self._last_update = now

        t = list(self.ts)
        for i, p in enumerate(self.plots):
            # lock X axis
            p.enableAutoRange('x', False)
            p.setXRange(t[0], t[-1], padding=0)

            # apply custom y‐limits if provided
            if self.ylims[i] is not None:
                ymin, ymax = self.ylims[i]
                p.setYRange(ymin, ymax, padding=0)
            else:
                p.enableAutoRange('y', True)

            # update curves
            for j, curve in enumerate(self.curves[i]):
                curve.setData(t, list(self.ys[i][j]))
