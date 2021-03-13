import random
import math

import constants
from constants import *


class FiniteStateMachine(object):
    """
    A finite state machine.
    """
    def __init__(self, state):
        self.state = state

    def change_state(self, new_state):
        self.state = new_state

    def update(self, agent):
        self.state.check_transition(agent, self)
        self.state.execute(agent)


class State(object):
    """
    Abstract state class.
    """
    def __init__(self, state_name):
        """
        Creates a state.

        :param state_name: the name of the state.
        :type state_name: str
        """
        self.state_name = state_name

    def check_transition(self, agent, fsm):
        """
        Checks conditions and execute a state transition if needed.

        :param agent: the agent where this state is being executed on.
        :param fsm: finite state machine associated to this state.
        """
        raise NotImplementedError("This method is abstract and must be implemented in derived classes")

    def execute(self, agent):
        """
        Executes the state logic.

        :param agent: the agent where this state is being executed on.
        """
        raise NotImplementedError("This method is abstract and must be implemented in derived classes")


class MoveForwardState(State):
    def __init__(self):
        super().__init__("MoveForward")
        # Todo: add initialization code
        self.t = 0
        self.v = constants.FORWARD_SPEED
        self.w = 0

    def check_transition(self, agent, state_machine):
        # Todo: add logic to check and execute state transition
        if agent.get_bumper_state():
            state_machine.change_state(GoBackState())
        elif self.t > constants.MOVE_FORWARD_TIME:
            state_machine.change_state(MoveInSpiralState())
        else:
            pass

    def execute(self, agent):
        # Todo: add execution logic
        agent.set_velocity(self.v, self.w)
        agent.move()
        self.t += constants.SAMPLE_TIME


class MoveInSpiralState(State):
    def __init__(self):
        super().__init__("MoveInSpiral")
        # Todo: add initialization code
        self.t = 0
        self.v = constants.FORWARD_SPEED
        self.w = constants.ANGULAR_SPEED

    def check_transition(self, agent, state_machine):
        # Todo: add logic to check and execute state transition
        if agent.get_bumper_state():
            state_machine.change_state(GoBackState())
        elif self.t > constants.MOVE_IN_SPIRAL_TIME:
            state_machine.change_state(MoveForwardState())
        else:
            pass

    def execute(self, agent):
        # Todo: add execution logic
        radius = constants.INITIAL_RADIUS_SPIRAL + SPIRAL_FACTOR * self.t
        self.w = self.v/radius
        agent.set_velocity(self.v, self.w)
        agent.move()
        self.t += constants.SAMPLE_TIME
        pass


class GoBackState(State):
    def __init__(self):
        super().__init__("GoBack")
        # Todo: add initialization code
        self.t = 0
        self.v = constants.BACKWARD_SPEED
        self.w = 0

    def check_transition(self, agent, state_machine):
        # Todo: add logic to check and execute state transition
        if self.t > constants.GO_BACK_TIME:
            state_machine.change_state(RotateState())
        else:
            pass

    def execute(self, agent):
        # Todo: add execution logic
        agent.set_velocity(self.v, self.w)
        agent.move()
        self.t += constants.SAMPLE_TIME


class RotateState(State):
    def __init__(self):
        super().__init__("Rotate")
        self.t = 0
        self.v = 0
        self.w = constants.ANGULAR_SPEED
        self.t_max = random.uniform(-math.pi/2, math.pi/2) / self.w

    def check_transition(self, agent, state_machine):
        # Todo: add logic to check and execute state transition
        if self.t > self.t_max:
            state_machine.change_state(MoveForwardState())
        else:
            pass
    
    def execute(self, agent):
        # Todo: add execution logic
        agent.set_velocity(self.v, self.w)
        agent.move()
        self.t += constants.SAMPLE_TIME

