from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import (
    Node, Problem,
)
from aimacode.utils import expr
from lp_utils import (
    FluentState, encode_state, decode_state,
)
from my_planning_graph import PlanningGraph

import itertools

class AirCargoProblem(Problem):
    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        """

        :param cargos: list of str
            cargos in the problem
        :param planes: list of str
            planes in the problem
        :param airports: list of str
            airports in the problem
        :param initial: FluentState object
            positive and negative literal fluents (as expr) describing initial state
        :param goal: list of expr
            literal fluents required for goal test
        """
        self.state_map = initial.pos + initial.neg
        self.initial_state_TF = encode_state(initial, self.state_map)
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        self.cargos = cargos
        self.planes = planes
        self.airports = airports
        self.actions_list = self.get_actions()

    def get_actions(self):
        '''
        This method creates concrete actions (no variables) for all actions in the problem
        domain action schema and turns them into complete Action objects as defined in the
        aimacode.planning module. It is computationally expensive to call this method directly;
        however, it is called in the constructor and the results cached in the `actions_list` property.

        Returns:
        ----------
        list<Action>
            list of Action objects
        '''

        def load_actions():
            '''Create all concrete Load actions and return a list

            :return: list of Action objects
            '''
            # generate a list of action objects for each permutation of (cargo, airplane, and airport)
                     # create load action object for cargo, airplane, airport
            loads = [Action(expr("Load({}, {}, {})".format(tup[0],tup[1],tup[2])),
                     # list of positive preconditions
                     [[expr("At({}, {})".format(tup[0],tup[2])),
                     expr("At({}, {})".format(tup[1],tup[2]))]
                     # list of negative preconditions
                     ,[]],
                     # list of add effects for cargo, airplane
                     [[expr("In({}, {})".format(tup[0],tup[1]))],
                     # list of negative effects for cargo, airport
                     [expr("At({}, {})".format(tup[0],tup[2]))]])
                     # for each permutation of (cargos, planes, airports)
                     for tup in list(itertools.product(self.cargos, self.planes, self.airports))]
            return loads

        def unload_actions():
            '''Create all concrete Unload actions and return a list

            :return: list of Action objects
            '''
            # generate a list of action objects for each permutation of (cargo, airplane, and airport)
                     # create load action object for cargo, airplane, airport
            unloads = [Action(expr("Unload({}, {}, {})".format(tup[0],tup[1],tup[2])),
                       # list of positive preconditions
                       [[expr("In({}, {})".format(tup[0], tup[1])),
                       expr("At({}, {})".format(tup[1],tup[2]))]
                       # list of negative preconditions
                       ,[]],
                       # list of add effects for cargo, airport
                       [[expr("At({}, {})".format(tup[0],tup[2]))],
                       # list of negative effects for cargo, airport
                       [expr("In({}, {})".format(tup[0],tup[1]))]])
                       # for each permutation of (cargos, planes, airports)
                       for tup in list(itertools.product(self.cargos, self.planes, self.airports))]
            return unloads

        def fly_actions():
            '''Create all concrete Fly actions and return a list

            :return: list of Action objects
            '''
            flys = []
            for fr in self.airports:
                for p in self.planes:
                    for to in self.airports:
                        if fr != to:
                            precond_pos = [expr("At({}, {})".format(p, fr)),
                                           ]
                            precond_neg = []
                            effect_add = [expr("At({}, {})".format(p, to))]
                            effect_rem = [expr("At({}, {})".format(p, fr))]
                            fly = Action(expr("Fly({}, {}, {})".format(p, fr, to)),
                                         [precond_pos, precond_neg],
                                         [effect_add, effect_rem])
                            flys.append(fly)
            return flys

        return load_actions() + unload_actions() + fly_actions()

    def actions(self, state: str) -> list:
        """ Return the actions that can be executed in the given state.

        :param state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'
        :return: list of Action objects
        """

        possible_actions = []
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())

        for action in self.actions_list:
            valid_pos = all([pc in kb.clauses for pc in action.precond_pos])
            valid_neg = all([pc not in kb.clauses for pc in action.precond_neg])
            if valid_pos and valid_neg: possible_actions.append(action)
        return possible_actions

    def result(self, state: str, action: Action):
        """ Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).

        :param state: state entering node
        :param action: Action applied
        :return: resulting state after action
        """
        new_state = FluentState([], [])
        old_state = decode_state(state, self.state_map)

        new_state.pos.extend(f for f in old_state.pos if f not in action.effect_rem)
        new_state.pos.extend(f for f in action.effect_add if f not in new_state.pos)
        new_state.neg.extend(f for f in old_state.neg if f not in action.effect_add)
        new_state.neg.extend(f for f in action.effect_rem if f not in new_state.neg)

        return encode_state(new_state, self.state_map)

    def goal_test(self, state: str) -> bool:
        """ Test the state to see if goal is reached

        :param state: str representing state
        :return: bool
        """
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for clause in self.goal:
            if clause not in kb.clauses:
                return False
        return True

    def h_1(self, node: Node):
        # note that this is not a true heuristic
        h_const = 1
        return h_const

    def h_pg_levelsum(self, node: Node):
        '''
        This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        '''
        # requires implemented PlanningGraph class
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum

    def h_ignore_preconditions(self, node: Node):
        '''
        This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.
        '''
        count = 0
        goals = set(self.goal)

        actions = self.actions_list[:]

        while(goals):
            next_node = max(actions, key=lambda x: len([eff for eff in x.effect_add if eff in goals]))
            actions.remove(next_node)
            goals= goals-set(next_node.effect_add)
            count+=1
        return count


def air_cargo_p1() -> AirCargoProblem:
    cargos = ['C1', 'C2']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
    neg = [expr('At(C2, SFO)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('At(C1, JFK)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('At(P1, JFK)'),
           expr('At(P2, SFO)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p2() -> AirCargoProblem:
    cargos = ['C1', 'C2', 'C3']
    planes = ['P1', 'P2', 'P3']
    airports = ['SFO', 'LAS', 'OAK']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, LAS)'),
           expr('At(C3, OAK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, LAS)'),
           expr('At(P3, OAK)'),
           ]
    neg = [

           expr('At(C1, LAS)'),
           expr('At(C1, OAK)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('In(C1, P3)'),

           expr('At(C2, SFO)'),
           expr('At(C2, OAK)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('In(C2, P3)'),

           expr('At(C3, SFO)'),
           expr('At(C3, LAS)'),
           expr('In(C3, P1)'),
           expr('In(C3, P2)'),
           expr('In(C3, P3)'),

           expr('At(P1, OAK)'),
           expr('At(P1, LAS)'),

           expr('At(P2, SFO)'),
           expr('At(P2, OAK)'),

           expr('At(P3, SFO)'),
           expr('At(P3, LAS)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, LAS)'),
            expr('At(C2, SFO)'),
            expr('At(C3, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p3() -> AirCargoProblem:
    cargos = ['C1', 'C2', 'C3', 'C4']
    planes = ['P1', 'P2']
    airports = ['OAK', 'SFO', 'LAS', 'LAX']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, OAK)'),
           expr('At(C3, LAS)'),
           expr('At(C4, LAX)'),
           expr('At(P1, SFO)'),
           expr('At(P2, OAK)'),
           ]
    neg = [expr('At(C1, OAK)'),
           expr('At(C1, LAS)'),
           expr('At(C1, LAX)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),

           expr('At(C2, SFO)'),
           expr('At(C2, LAS)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),

           expr('At(C3, SFO)'),
           expr('At(C3, OAK)'),
           expr('At(C3, LAX)'),
           expr('In(C3, P1)'),
           expr('In(C3, P2)'),

           expr('At(C4, SFO)'),
           expr('At(C4, OAK)'),
           expr('At(C4, LAS)'),
           expr('In(C4, P1)'),
           expr('In(C4, P2)'),

           expr('At(P1, OAK)'),
           expr('At(P1, LAS)'),
           expr('At(P1, LAX)'),

           expr('At(P2, SFO)'),
           expr('At(P2, LAS)'),
           expr('At(P2, LAX)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, OAK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, OAK)'),
            expr('At(C4, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)
