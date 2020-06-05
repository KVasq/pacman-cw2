# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py
# learningAgents.py

from pacman import Directions
from game import Agent
import random
import game
import util

# QLearnAgent
#
class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.35, epsilon=0.1, gamma=0.8, numTraining = 10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0

        #An array for current state (index = 0) and next state (index = 1)
        self.states = [None] * 2

        #An array for last action (index = 0) and next action (index = 1)
        self.actions = [None] * 2

        #A dictionary which holds a Q-value per (state, action) input
        self.Qvalues = {}

    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value

    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts

    def updateQvalue(self, state):
        #Checks the state and action inputs and assigns a defualt of 0 if they don't have an entry in the Q value table
        if (self.states[1], self.actions[1]) not in self.Qvalues.keys():
            self.Qvalues[(self.states[1], self.actions[1])] = 0
        if(self.states[0], self.actions[0]) not in self.Qvalues.keys():
            self.Qvalues[(self.states[0], self.actions[0])] = 0

        #The reward is the change of score that the agent achieves with the change of state
        try:
            reward = self.states[1].getScore() - self.states[0].getScore()
        except AttributeError:
            reward = state.getScore() - 0

        #Updates the Q value entry using the Q-learing algorithm
        temporalDifference = reward + self.gamma * self.Qvalues.get((self.states[1], self.argMax(self.states[1], self.actions[1]))) - self.Qvalues.get((self.states[0], self.actions[0]))
        self.Qvalues[(self.states[0], self.actions[0])] = self.Qvalues.get((self.states[0], self.actions[0])) + self.alpha * temporalDifference

    def argMax(self, state, randAction):
        #the default action is the randomly assigned one
        direction = randAction

        #takes the Qvalue of the default state-action pair to compare against others
        value = self.Qvalues.get((state, direction))

        #Iterates all Q-value possibilities and choosing the highest one
        for a in state.getLegalPacmanActions():
            if a != Directions.STOP:
                try:
                    if self.Qvalues.get((state, a)) > value:
                        value = self.Qvalues.get((state, a))
                        direction = a
                except KeyError:
                    self.Qvalues[(state, a)] = 0

        #returns an action
        return direction

    # getAction
    #
    # The main method required by the game. Called every time that
    # Pacman is expected to move
    def getAction(self, state):

        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        print "Legal moves: ", legal
        print "Pacman position: ", state.getPacmanPosition()
        print "Ghost positions:" , state.getGhostPositions()
        print "Food locations: "
        #print state.getFood()
        print "Score: ", state.getScore()

        #Pass s' (next state) to s (current state) and assign s' the game state
        self.states[0] = self.states[1]
        self.states[1] = state

        # Now pick what action to take. For now a random choice among
        # the legal moves
        pick = random.choice(legal)

        #Pass a' (next action) to a (current action) and assign a' a random legal action
        self.actions[0] = self.actions[1]
        self.actions[1] = pick

        #Assigns the next action to be that which achieves the highest Q-valued state
        if random.random() > self.epsilon:
            pick = self.argMax(self.states[1], self.actions[1])

        self.actions[1] = pick

        self.updateQvalue(state)

        # We have to return an action
        return pick

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):

        print "A game just ended!"

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes

        self.states[0] = self.states[1]
        self.states[1] = state

        self.actions[0] = self.actions[1]
        self.actions[1] = Directions.STOP

        self.updateQvalue(state)

        #resets action to its initial game start value
        self.actions[1] = 'None'

        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)
