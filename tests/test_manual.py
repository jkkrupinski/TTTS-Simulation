import curses
import gymnasium as gym

import environment


def main(stdscr):
    env = gym.make("camera-v3", render_mode="human")

    observations = env.reset()[0]

    run = True
    action = 0

    stdscr.addstr("Press the arrow keys to move or 'q' to quit.\n")
    stdscr.refresh()

    while run:
        key = stdscr.getch()  # Wait for user input

        if key == curses.KEY_UP:
            action = 0
        elif key == curses.KEY_DOWN:
            action = 1
        elif key == curses.KEY_LEFT:
            action = 2
        elif key == curses.KEY_RIGHT:
            action = 3
        elif key == ord("q"):
            stdscr.addstr("Exiting...\n")
            run = False
            break

        observations, reward, terminated, truncated, info = env.step(action)


if __name__ == "__main__":
    curses.wrapper(main)
