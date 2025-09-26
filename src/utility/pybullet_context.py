from pybullet import connect, disconnect, GUI, DIRECT

class PyBulletContext:
   def __init__(self, graphics):
      self.__graphics = graphics

   def __enter__(self):
      connect({
         True: GUI,
         False: DIRECT
      }[self.__graphics])

      return self

   def __exit__(self, _type, _value, _traceback):
      disconnect()
