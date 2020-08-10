# searchAgents.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
This file contains all of the agents that can be selected to 
control Pacman.  To select an agent, use the '-p' option
when running pacman.py.  Arguments can be passed to your agent
using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a searchFunction=depthFirstSearch

Commands to invoke other search strategies can be found in the 
project description.

Please only change the parts of the file you are asked to.
Look for the lines that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the
project description for details.

Good luck and happy searching!
"""
from game import Directions
from game import Agent
from game import Actions
import util
import time
import search
import searchAgents

class GoWestAgent(Agent):
  "An agent that goes West until it can't."
  
  def getAction(self, state):
    "The agent receives a GameState (defined in pacman.py)."
    if Directions.WEST in state.getLegalPacmanActions():
      return Directions.WEST
    else:
      return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
  """
  This very general search agent finds a path using a supplied search algorithm for a
  supplied search problem, then returns actions to follow that path.
  
  As a default, this agent runs DFS on a PositionSearchProblem to find location (1,1)
  
  Options for fn include:
    depthFirstSearch or dfs
    breadthFirstSearch or bfs
    
  
  Note: You should NOT change any code in SearchAgent
  """
    
  def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
    # Warning: some advanced Python magic is employed below to find the right functions and problems
    
    # Get the search function from the name and heuristic
    if fn not in dir(search): 
      raise AttributeError, fn + ' is not a search function in search.py.'
    func = getattr(search, fn)
    if 'heuristic' not in func.func_code.co_varnames:
      print('[SearchAgent] using function ' + fn) 
      self.searchFunction = func
    else:
      if heuristic in dir(searchAgents):
        heur = getattr(searchAgents, heuristic)
      elif heuristic in dir(search):
        heur = getattr(search, heuristic)
      else:
        raise AttributeError, heuristic + ' is not a function in searchAgents.py or search.py.'
      print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic)) 
      # Note: this bit of Python trickery combines the search algorithm and the heuristic
      self.searchFunction = lambda x: func(x, heuristic=heur)
      
    # Get the search problem type from the name
    if prob not in dir(searchAgents) or not prob.endswith('Problem'): 
      raise AttributeError, prob + ' is not a search problem type in SearchAgents.py.'
    self.searchType = getattr(searchAgents, prob)
    print('[SearchAgent] using problem type ' + prob) 
    
  def registerInitialState(self, state):
    """
    This is the first time that the agent sees the layout of the game board. Here, we
    choose a path to the goal.  In this phase, the agent should compute the path to the
    goal and store it in a local variable.  All of the work is done in this method!
    
    state: a GameState object (pacman.py)
    """
    if self.searchFunction == None: raise Exception, "No search function provided for SearchAgent"
    starttime = time.time()
    problem = self.searchType(state) # Makes a new search problem
    self.actions  = self.searchFunction(problem) # Find a path
    totalCost = problem.getCostOfActions(self.actions)

    # ATTENTION: we changed code here
    print('Path found with total cost of %d in %.6f seconds' % (totalCost, time.time() - starttime))
    if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)
    
  def getAction(self, state):
    """
    Returns the next action in the path chosen earlier (in registerInitialState).  Return
    Directions.STOP if there is no further action to take.
    
    state: a GameState object (pacman.py)
    """
    if 'actionIndex' not in dir(self): self.actionIndex = 0
    i = self.actionIndex
    self.actionIndex += 1
    if i < len(self.actions):
      return self.actions[i]    
    else:
      return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
  """
  A search problem defines the state space, start state, goal test,
  successor function and cost function.  This search problem can be 
  used to find paths to a particular point on the pacman board.
  
  The state space consists of (x,y) positions in a pacman game.
  
  Note: this search problem is fully specified; you should NOT change it.
  """
  
  def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True):
    """
    Stores the start and goal.  
    
    gameState: A GameState object (pacman.py)
    costFn: A function from a search state (tuple) to a non-negative number
    goal: A position in the gameState
    """
    self.walls = gameState.getWalls()
    self.startState = gameState.getPacmanPosition()
    if start != None: self.startState = start
    self.goal = goal
    self.costFn = costFn
    if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
      print 'Warning: this does not look like a regular search maze'

    # For display purposes
    self._visited, self._visitedlist, self._expanded = {}, [], 0

  def getStartState(self):
    return self.startState

  def isGoalState(self, state):
     isGoal = state == self.goal 
     
     # For display purposes only
     if isGoal:
       self._visitedlist.append(state)
       import __main__
       if '_display' in dir(__main__):
         if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
           __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable
       
     return isGoal   
   
  def getSuccessors(self, state):
    """
    Returns successor states, the actions they require, and a cost of 1.
    
     As noted in search.py:
         For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
    """
    
    successors = []
    for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
      x,y = state
      dx, dy = Actions.directionToVector(action)
      nextx, nexty = int(x + dx), int(y + dy)
      if not self.walls[nextx][nexty]:
        nextState = (nextx, nexty)
        cost = self.costFn(nextState)
        successors.append( ( nextState, action, cost) )
        
    # Bookkeeping for display purposes
    self._expanded += 1 
    if state not in self._visited:
      self._visited[state] = True
      self._visitedlist.append(state)
      
    return successors

  def getCostOfActions(self, actions):
    """
    Returns the cost of a particular sequence of actions.  If those actions
    include an illegal move, return 999999
    """
    if actions == None: return 999999
    x,y= self.getStartState()
    cost = 0
    for action in actions:
      # Check figure out the next state and see whether its' legal
      dx, dy = Actions.directionToVector(action)
      x, y = int(x + dx), int(y + dy)
      if self.walls[x][y]: return 999999
      cost += self.costFn((x,y))
    return cost

class StayEastSearchAgent(SearchAgent):
  """
  An agent for position search with a cost function that penalizes being in
  positions on the West side of the board.  
  
  The cost function for stepping into a position (x,y) is 1/2^x.
  """
  def __init__(self):
      self.searchFunction = search.uniformCostSearch
      costFn = lambda pos: .5 ** pos[0] 
      self.searchType = lambda state: PositionSearchProblem(state, costFn)
      
class StayWestSearchAgent(SearchAgent):
  """
  An agent for position search with a cost function that penalizes being in
  positions on the East side of the board.  
  
  The cost function for stepping into a position (x,y) is 2^x.
  """
  def __init__(self):
      self.searchFunction = search.uniformCostSearch
      costFn = lambda pos: 2 ** pos[0] 
      self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
  "The Manhattan distance heuristic for a PositionSearchProblem"
  xy1 = position
  xy2 = problem.goal
  return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
  "The Euclidean distance heuristic for a PositionSearchProblem"
  xy1 = position
  xy2 = problem.goal
  return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print 'Warning: no food in corner ' + str(corner)
        self._expanded = 0 # DO NOT CHANGE; Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem
        "*** YOUR CODE HERE ***"
        self.costFn = lambda x,y: 1
        # initialized corner visited state
        self.cornerVisited=[False,False,False,False]
        # define a state as (position,corner visited list)
        self.startState=(self.startingPosition,self.cornerVisited)

    def getStartState(self):
        """
        Returns the start state (in your state space, not the full Pacman state
        space)
        """
        "*** YOUR CODE HERE ***"
        return self.startState # self.startState=(self.startingPosition,self.cornerVisited)

    def isGoalState(self, state):
        """
        Returns whether this search state is a goal state of the problem.
        """
        "*** YOUR CODE HERE ***"
        isGoal = False
        cornerVisited = state[1]
        if cornerVisited == [True,True,True,True]:
            isGoal = True
        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
            For a given state, this should return a list of triples, (successor,
            action, stepCost), where 'successor' is a successor to the current
            state, 'action' is the action required to get there, and 'stepCost'
            is the incremental cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            # Here's a code snippet for figuring out whether a new position hits a wall:
            #   x,y = currentPosition
            #   dx, dy = Actions.directionToVector(action)
            #   nextx, nexty = int(x + dx), int(y + dy)
            #   hitsWall = self.walls[nextx][nexty]

            "*** YOUR CODE HERE ***"
            # current state
            x,y = state[0]
            cornerVisited = state[1]
            # direction
            dx,dy = Actions.directionToVector(action)
            # new position
            x_new,y_new = int(x + dx),int(y + dy)
            # copy the 'center' point's corner visited list,note this is a deep copy since cornerVisited is a variable
            cornerVisited_new = cornerVisited[:]
            isWall = self.walls[x_new][y_new]
            if not isWall:
                cornerIndex = 0 # index of each corresponding corner in the corner list
                # check if on of the corners is visited
                for corner in self.corners:
                    if (x_new,y_new) == corner:
                        break
                    cornerIndex += 1

                if cornerIndex < 4: # is in one corner
                    cornerVisited_new[cornerIndex] = True # update the corner visited list
                state_new = ((x_new,y_new),cornerVisited_new)
                cost = self.costFn(x_new,y_new)
                successors.append((state_new,action,cost)) # add this successor

        self._expanded += 1 # DO NOT CHANGE
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x,y= self.startingPosition # from the starting point
        cost = 0
        for action in actions:
            dx, dy = Actions.directionToVector(action) # direction of the action
            x, y = int(x + dx), int(y + dy) # new position
            if self.walls[x][y]: return 999999 # hit wall
            cost += self.costFn(x,y) # add the cost on this new position
        return cost

def cornersHeuristic(state, problem):
  """
  A heuristic for the CornersProblem that you defined.
  
    state:   The current search state 
             (a data structure you chose in your search problem)
    
    problem: The CornersProblem instance for this layout.  
    
  This function should always return a number that is a lower bound
  on the shortest path from the state to a goal of the problem; i.e.
  it should be admissible.  (You need not worry about consistency for
  this heuristic to receive full credit.)
  """
  corners = problem.corners # These are the corner coordinates
  walls = problem.walls # These are the walls of the maze, as a Grid (game.py)


  # THIS IS A NEW CODE BY OLEKSII & HYUNGJOON

  # Doesn't use information about walls, fast heuristic
  # RESULTS: < 1000
  res = simpleBestManhattanPath([item for item in corners if item not in state[1]], state[0])

  '''
  #RESULTS: approx. 800
  #Requires pre-calculation that can be stored in the problem object
  #(e.g. self.obs = calcMaxWalls(self.walls) during problem initialization).
  res = enhancedBestManhattanPath([item for item in corners if item not in state[1]], state[0], walls, problem.obs)
  '''

  # TODO: what's about an idea of some random to distinguish similar pathes?
  return res

# best manhattan path
def simpleBestManhattanPath(corners, pos):
    '''
    This function calculates the best path that starts at current position
    and go through all not visited yet corners. Implementation is just a recursive
    brute force. Overall it has a constant time (variants - only 1 starting position and 4 corners).
    '''

    if len(corners) == 0:
        return 0

    res = []
    for xy in corners:
        dist = abs(xy[0] - pos[0]) + abs(xy[1] - pos[1])
        dist = dist + simpleBestManhattanPath([c for c in corners if c != xy], xy)
        res.append(dist)

    return min(res)

# THIS IS A NEW CODE BY OLEKSII & HYUNGJOON
def enhancedBestManhattanPath(corners, pos, walls, obstacles):
    '''
    Still admissible, but maybe not consistent.
    Reults in less nodes expanded,
    but takes some time for pre-calculation.
    '''
    if len(corners) == 0:
        return 0

    top, right = walls.height-2, walls.width-2
    nominals = ((1,1), (1,top), (right, 1), (right, top))

    res = []
    for xy in corners:
        dist = abs(xy[0] - pos[0]) + abs(xy[1] - pos[1])

        # Additional idea: two walls near current position can for sure increase minimal path by the diagonal
        if xy[0] - pos[0] > 0 and xy[1] - pos[1] > 0 and walls[pos[0]+1][pos[1]] and walls[pos[0]][pos[1]+1]:
            dist = dist + 2
        elif xy[0] - pos[0] > 0 and xy[1] - pos[1] < 0 and walls[pos[0]+1][pos[1]] and walls[pos[0]][pos[1]-1]:
            dist = dist + 2
        elif xy[0] - pos[0] < 0 and xy[1] - pos[1] < 0 and walls[pos[0]-1][pos[1]] and walls[pos[0]][pos[1]-1]:
            dist = dist + 2
        elif xy[0] - pos[0] < 0 and xy[1] - pos[1] > 0 and walls[pos[0]-1][pos[1]] and walls[pos[0]][pos[1]+1]:
            dist = dist + 2

        # Additional idea: and not onlu by a diagonal
        if pos not in nominals:
            if xy[0] - pos[0] > 0 and xy[1] - pos[1] == 0 and walls[pos[0]+1][pos[1]]:
                dist = dist + 2
            elif xy[0] - pos[0] == 0 and xy[1] - pos[1] > 0 and walls[pos[0]][pos[1]+1]:
                dist = dist + 2
            elif xy[0] - pos[0] < 0 and xy[1] - pos[1] == 0 and walls[pos[0]-1][pos[1]]:
                dist = dist + 2
            elif xy[0] - pos[0] == 0 and xy[1] - pos[1] < 0 and walls[pos[0]][pos[1]-1]:
                dist = dist + 2

        # Additional idea - let's calculate a path between corners (by a line)
        # using pre-calculated peaks
        if pos == (1,1) and xy == (right, 1) or pos == (right,1) and xy == (1, 1):
            dist = dist + obstacles[0]
        if pos == (right, 1) and xy == (right, top) or pos == (right, top) and xy == (right, 1):
            dist = dist + obstacles[1]
        if pos == (right, top) and xy == (1,top) or pos == (1,top) and xy == (right, top):
            dist = dist + obstacles[2]
        if pos == (1,1) and xy == (1,top) or pos == (1,top) and xy == (1, 1):
            dist = dist + obstacles[3]

        dist = dist + enhancedBestManhattanPath([c for c in corners if c != xy], xy, walls, obstacles)
        res.append(dist)
    return min(res)

# THIS IS A NEW CODE BY OLEKSII & HYUNGJOON
def calcMaxWalls(walls):
    '''
    The current function calculates heights of max wall peaks
    between each corner adjacent to the same side of a map.
    It adds peak * 2 to the smallest path between them.
    Straightforward brute force implementation O(n^2)
    '''

    top, right = walls.height-2, walls.width-2

    # assume path always exists
    resTop = []
    for i in range(2, right):
        if walls[i][1]:
            cnt = 1
            for j in range (2, top):
                if walls[i][j]:
                    cnt = cnt + 1
            resTop.append(cnt)

    resBottom = []
    for i in range(2, right):
        if walls[i][top]:
            cnt = 1
            for j in range (top-1, 1):
                if walls[i][j]:
                    cnt = cnt + 1
            resBottom.append(cnt)

    resLeft = []
    for i in range(2, top):
        if walls[1][i]:
            cnt = 1
            for j in range (2, right):
                if walls[j][i]:
                    cnt = cnt + 1
            resLeft.append(cnt)

    resRight = []
    for i in range(2, top):
        if walls[right][i]:
            cnt = 1
            for j in range (right-1, 1):
                if walls[i][j]:
                    cnt = cnt + 1
            resRight.append(cnt)

    return (max(resTop) * 2, max(resRight) * 2, max(resBottom) * 2, max(resLeft) * 2)

class AStarCornersAgent(SearchAgent):
  "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
  def __init__(self):
    self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
    self.searchType = CornersProblem

class FoodSearchProblem:
  """
  A search problem associated with finding the a path that collects all of the 
  food (dots) in a Pacman game.
  
  A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
    pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
    foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food 
  """
  def __init__(self, startingGameState):
    self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
    self.walls = startingGameState.getWalls()
    self.startingGameState = startingGameState
    self._expanded = 0
    self.heuristicInfo = {} # A dictionary for the heuristic to store information
      
  def getStartState(self):
    return self.start
  
  def isGoalState(self, state):
    return state[1].count() == 0

  def getSuccessors(self, state):
    "Returns successor states, the actions they require, and a cost of 1."
    successors = []
    self._expanded += 1
    for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
      x,y = state[0]
      dx, dy = Actions.directionToVector(direction)
      nextx, nexty = int(x + dx), int(y + dy)
      if not self.walls[nextx][nexty]:
        nextFood = state[1].copy()
        nextFood[nextx][nexty] = False
        successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
    return successors

  def getCostOfActions(self, actions):
    """Returns the cost of a particular sequence of actions.  If those actions
    include an illegal move, return 999999"""
    x,y= self.getStartState()[0]
    cost = 0
    for action in actions:
      # figure out the next state and see whether it's legal
      dx, dy = Actions.directionToVector(action)
      x, y = int(x + dx), int(y + dy)
      if self.walls[x][y]:
        return 999999
      cost += 1
    return cost

class AStarFoodSearchAgent(SearchAgent):
  "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
  def __init__(self):
    self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
    self.searchType = FoodSearchProblem

def foodHeuristic(state, problem):
  """
  Your heuristic for the FoodSearchProblem goes here.
  
  This heuristic must be consistent to ensure correctness.  First, try to come up
  with an admissible heuristic; almost all admissible heuristics will be consistent
  as well.
  
  If using A* ever finds a solution that is worse uniform cost search finds,
  your heuristic is *not* consistent, and probably not admissible!  On the other hand,
  inadmissible or inconsistent heuristics may find optimal solutions, so be careful.
  
  The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a 
  Grid (see game.py) of either True or False. You can call foodGrid.asList()
  to get a list of food coordinates instead.
  
  If you want access to info like walls, capsules, etc., you can query the problem.
  For example, problem.walls gives you a Grid of where the walls are.
  
  If you want to *store* information to be reused in other calls to the heuristic,
  there is a dictionary called problem.heuristicInfo that you can use. For example,
  if you only want to count the walls once and store that value, try:
    problem.heuristicInfo['wallCount'] = problem.walls.count()
  Subsequent calls to this heuristic can access problem.heuristicInfo['wallCount']
  """
  position, foodGrid = state

  # THIS IS A NEW CODE BY OLEKSII & HYUNGJOON
  '''
  This HEURISTIC is for sure admissible and consistent.
  The relaxed representation operates with the smallest possible path -
  if all left dots are accessible in a line from the current position
  So we can't get an overestimation. At the same time - next state
  can be 1 dot less or just equal - consistency is kept.
  RESULTS: 8679 nodes
  '''

  # Just so simple :)
  return len(foodGrid.asList())

  '''
  # POSSIBLE OPTIMIZATION OF THE PREVIOUS IDEA
  # trying to calculate number of gaps in a line of left dots
  # that will be for sure
  res = len(foodGrid.asList())

  cnt = 0
  for (x, y) in foodGrid.asList():
      #if isAlone(x, y, foodGrid):
      cnt = cnt + isAlone(x, y, foodGrid)

  return res + cnt/2
  '''

  '''
  # ANOTHER APPROACH - MANHATTAN PATH ALONG CONVEX HULL
  # Admissible but possible inconsistent heuristic,
  # some modifications needed to make it consistent,
  # like taking into account a current position

  l = foodGrid.asList()
  #l.append(position)
  hull = convex_hull(l)

  #print (position)
  #print (hull)
  #print (">")
  #print ()

  #ind = hull.index(position)

  # Base cases
  if len(hull) == 0:
      return 0
  if len(hull) == 1:
      return 1
  if len(hull) == 2:
      return abs(hull[0][0] - hull[1][0]) + abs(hull[0][1] - hull[1][0])

  res = 0
  maxDist = -1
  hull.append(hull[0])
  for i in range(0, len(hull)-1):
      # Again using manhattan distances
      dist = abs(hull[i][0] - hull[i+1][0]) + abs(hull[i][1] - hull[i+1][0])
      if maxDist < dist:
          # Removing the longest segment, because p could be closer
          maxDist = dist
      res = res + dist
  res = res - maxDist

  return res
  '''


# THIS IS A CODE ADDED BY OLEKSII & HYUNGJOON FROM
def isAlone(x, y, foodGrid):
    # Improvement for min line heuristic
    if foodGrid[x+1][y] or foodGrid[x][y+1]:
        return 0
    if foodGrid[x-1][y] or foodGrid[x][y-1]:
        return 0
    return 1

# THIS IS A CODE ADDED BY OLEKSII & HYUNGJOON FROM
# http://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain
def convex_hull(points):
    """Computes the convex hull of a set of 2D points.
    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list.
    return lower[:-1] + upper[:-1]


  
class ClosestDotSearchAgent(SearchAgent):
  "Search for all food using a sequence of searches"
  def registerInitialState(self, state):
    self.actions = []
    currentState = state
    while(currentState.getFood().count() > 0): 
      nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
      self.actions += nextPathSegment
      for action in nextPathSegment: 
        legal = currentState.getLegalActions()
        if action not in legal: 
          t = (str(action), str(currentState))
          raise Exception, 'findPathToClosestDot returned an illegal move: %s!\n%s' % t
        currentState = currentState.generateSuccessor(0, action)
    self.actionIndex = 0
    print 'Path found with cost %d.' % len(self.actions)
    
  def findPathToClosestDot(self, gameState):
    "Returns a path (a list of actions) to the closest dot, starting from gameState"
    # Here are some useful elements of the startState
    startPosition = gameState.getPacmanPosition()
    food = gameState.getFood()
    walls = gameState.getWalls()
    problem = AnyFoodSearchProblem(gameState)

    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
  
class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state

        x,y = state
        return self.food[x][y]==True

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))