from .base import Base
from threading import Thread
from ..utility import PyBulletContext
from time import sleep
from os import environ

time_step = int(environ.get("TIME_STEP", "240"))

with PyBulletContext(True):
    base = Base(time_step=time_step)

    def handle_input():
        while True:
            message = input("> ")

            if message == "exit":
                break
            elif message == "break":
                breakpoint()
                continue
            elif message == "reset":
                base.reset()
                continue
            else:
                base.evaluate(message)

    thread = Thread(target=handle_input)
    thread.start()

    while thread.is_alive():
        base.update()
        sleep(1.0 / time_step)
