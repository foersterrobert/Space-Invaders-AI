from game import SpaceInvadersAI
import itertools
from agent import Agent

def main():
    game = SpaceInvadersAI()
    agent = Agent()
    record = 0
    run = True
    for step in itertools.count():
        if not run:
            break
        state_old = game.get_state()
        action = agent.get_action(state_old, step)
        reward, done, score = game.check_events(action, True)
        state_new = game.get_state()
        agent.train((state_old, action, reward, state_new, done), step)
        if done:
            if score > record:
                record = score
                agent.save()
                print('Record:', record)

if __name__ == '__main__':
    main()