import random
import math
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

    def check_transition(self, agent, state_machine):
        # Todo: add logic to check and execute state transition
        pass

    def execute(self, agent):
        # Todo: add execution logic
        pass


class MoveInSpiralState(State):
    def __init__(self):
        super().__init__("MoveInSpiral")
        # Todo: add initialization code
    
    def check_transition(self, agent, state_machine):
        # Todo: add logic to check and execute state transition
        pass

    def execute(self, agent):
        # Todo: add execution logic
        pass


class GoBackState(State):
    def __init__(self):
        super().__init__("GoBack")
        # Todo: add initialization code

    def check_transition(self, agent, state_machine):
        # Todo: add logic to check and execute state transition
        pass

    def execute(self, agent):
        # Todo: add execution logic
        pass


class RotateState(State):
    def __init__(self):
        super().__init__("Rotate")
        # Todo: add initialization code

    def check_transition(self, agent, state_machine):
        # Todo: add logic to check and execute state transition
        pass
    
    def execute(self, agent):
        # Todo: add execution logic
        pass
