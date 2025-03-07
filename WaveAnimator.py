from matplotlib import pyplot as plt 
import matplotlib as mpl
import numpy as np 
from matplotlib.animation import FuncAnimation, FFMpegWriter

class WaveAnimator:
  def __init__(self, x, t, ylim=(-3,3)):

    self.fig = plt.figure()
    xLow = xGrid[0]
    xHigh = xGrid[len(xGrid)-1]

    self.axis = plt.axes(xlim=(xLow, xHigh), ylim=ylim)
    self.pLine = self.axis.plot([], [], 'b-')

    self.uData = []

  def run(self, uData):

    self.uData = uData

    def animate(i):
      pData_i = self.uData[i]
      self.pLine.set_ydata(pData_i)

      return self.pLine
    

    ani = FuncAnimation(self.fig, animate, frames=len(self.uData),
                          blit=True)
    
    plt.show()
    

    
if __name__=='__main__':
  xGrid = np.linspace(0.0, 1.0, 101)

  tMax = 10.0
  tGrid = np.linspace(0.0, tMax, 101)
  
  uData = []
  times = []
  for t in tGrid:
    times.append(t)
    p = np.cos(2 * np.pi * (t - 4.0 * xGrid))

    uData.append((p))

  xLow = xGrid[0]
  xHigh = xGrid[len(xGrid)-1]

  fig = plt.figure()

  axis = plt.axes(xlim=(xLow, xHigh), ylim=(-1.5,1.5))
  #w = WaveAnimator(xGrid)

  pLine, = axis.plot(xGrid, uData[0][0], 'b-')

  def animate(i):
    axis.set_title('time={:.4f}'.format(times[i]))
    pData_i = uData[i]
    pLine.set_ydata(pData_i)

    return pLine

  #mpl.use('Agg')

  ani = FuncAnimation(fig, animate, frames=len(uData),
                          blit=False)  
  
  writervideo = FFMpegWriter(fps=10) 
  
  plt.show()

  #ani.save('movie.mp4', writer=writervideo)
  #w.run(uData)