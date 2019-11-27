import math
from simpleai.search import astar, SearchProblem, depth_first, breadth_first, iterative_limited_depth_first, idastar

# Class containing the methods to solve the maze
class MazeSolver(SearchProblem):
    # Initialize the class 
    def __init__(self, board):
        self.board = board
        self.goal = (0, 0)

        for y in range(len(self.board)):
            for x in range(len(self.board[y])):
                if self.board[y][x].lower() == "o":
                    self.initial = (x, y)
                elif self.board[y][x].lower() == "x":
                    self.goal = (x, y)

        super(MazeSolver, self).__init__(initial_state=self.initial)

    # Define the method that takes actions
    # to arrive at the solution    액션은 벽에 걸리지 않게 핸들링해주는 부분을 포함하고 있음.
    def actions(self, state):
        actions = []
        for action in COSTS.keys():
            newx, newy = self.result(state, action)
            if self.board[newy][newx] != "#":
                actions.append(action)

        return actions

    # Update the state based on the action  result는 state와 action을 받아서 새로운 state를 만들어낸다.
    def result(self, state, action):
        x, y = state

        if action.count("up"):
            y -= 1
        if action.count("down"):
            y += 1
        if action.count("left"):
            x -= 1
        if action.count("right"):
            x += 1

        new_state = (x, y)

        return new_state

    # Check if we have reached the goal 목표점에 도달했는지 체크해주는 부분.
    def is_goal(self, state):
        return state == self.goal

     #Compute the cost of taking an action 우리 과제에선 코스트 상관할 필요 전혀 없음.
    def cost(self, state, action, state2):
        return COSTS[action]


    # Heuristic that we use to arrive at the solution 피타고라스 정의로 직선거리를 잰거라고 함.
    def heuristic(self, state):
        x, y = state
        gx, gy = self.goal

        return math.sqrt((x - gx) ** 2 + (y - gy) ** 2)


#여기서 부터 실제 코드가 실행되는 부분이다!!! 맵은 직접 손으로 그렸고, o와 x를 problem클래스에게 인자로 전달함으로써
#problem클래스는 이 인자들을 읽어들이면서 맵을 알아낸다.

if __name__ == "__main__":
    # Define the map
    MAP = """
    ##############################
    #         #              #   #
    # ####    ########       #   #
    #  o #    #              #   #
    #    ###     #####  ######   #
    #      #   ###   #           #
    #      #     #   #  #  #   ###
    #     #####    #    #  # x   #
    #              #       #     #
    ##############################
    """

    # Convert map to a list    프린트(맵)으로 맵을 한번 출력해줌.
    print(MAP)
    MAP = [list(x) for x in MAP.split("\n") if x]


    # Define cost of moving around the map 코스트는 우리과제랑 전혀 상관없음 그러니까 신경쓰지마!!
    cost_regular = 1.0
    cost_diagonal = 1.7

    # Create the cost dictionary
    COSTS = {
        "up": cost_regular,
        "down": cost_regular,
        "left": cost_regular,
        "right": cost_regular,
        "up left": cost_diagonal,
        "up right": cost_diagonal,
        "down left": cost_diagonal,
        "down right": cost_diagonal,
    }

    # Create maze solver object   메이즈솔버 클래스에 맵을 파라미터로 넘김. 이제 문제를 생성시킨거임.
    problem = MazeSolver(MAP)

    # Run the solver 여기에서 실행 주석해가면서 전부다 넣어서 실행시켜보자. 그리고 그에 따른 결과를 출력ㄱㄱ
    # 다 임포트 시킬것. simpleai-search-tranditinal안에 함수들이 구현되어 있음.
    #저기 가보면 코드가 구현되어 있는것이 아니라 그냥 이미 구현되어 있는것(search)을 불러내는 방식으로 되어 있음.
    #search 함수에 있는 파라미터를 조작함으로써 과제의 원하는 값을 얻어낼 수 있음.
    #예를 들면, 브리드 퍼스트에선 피포리스트를 사용하고, 프라블럼도 인자로 넘기고, 그래프 서치, 뷰 등등(뒤에 2개는 자세히 설명 x)
    #depth first같은 경우에는 라스트인 퍼스트아웃 리스트를 사용,
    #여기다가 ida* 구현해. 이름적고 return 값적고, 핸들링 하는 부분 어디다 추가하는진 모르겠지만 그것만 잘 추가하면 될것이야~(웬지 맨 밑에 search일것 같은데...)
    #memory라는 set이 있음. set이 뭐냐! 탐색했던 노드들을 메모리에 집어 넣는 것이다. 노드들을 탐색할때마다 메모리에다 에드를 시킴. 그렇다면, 메모리 렝스를 출력하면, 내가 탐색했던 노드의 길이수를 알 수 있겠지!
    #끝나기 전에 메모리 렝스를 출력하면 확장노드가 7개인것을 얻어낼 수 있음. 어디에다가 넣는진 모르겠지만, str(len(memory))를 잘 넣어봐.
    #또한, 생성노드는 fringe라는 리스트를 이용하면 구할 수 있음. fringe는 탐색할 노드들을 집어넣음. 종료직전에는 아직 탐색하지 않은 노드들만 남아있음.
    #ida스타는 a스타하고 iterative limited ~하고 합치면 된다.
    #f limit는 직접구현해야할거야~~ 평가함수값을 계산하는것도 다 나와있어 잘 찾아봐.
    #프로젝트 파일 전체하고 한글파일.


    #result = astar(problem, graph_search=True)
    #result = depth_first(problem, graph_search=True)
    #result = breadth_first(problem, graph_search=True)
    result = iterative_limited_depth_first(problem, graph_search=True)
    #result = idastar(problem, graph_search=True)
    # Extract the path
    path = [x[1] for x in result.path()]
    print("해길이", path)
    print(len(path))

    # Print the result
    print()
    for y in range(len(MAP)):
        for x in range(len(MAP[y])):
            if (x, y) == problem.initial:
                print('o', end='')
            elif (x, y) == problem.goal:
                print('x', end='')
            elif (x, y) in path:
                print('·', end='')
            else:
                print(MAP[y][x], end='')

        print()

