import numpy as np

class Drone_simulator:    
    def __init__(self, sim, polygon, scaling_factor, height, num_robots=3):
        self.scaling_factor = scaling_factor
        self.scaled_polygon = [(x/scaling_factor,y/scaling_factor) for (x,y) in polygon]
        self.rounded_polygon = self.scaled_polygon + [self.scaled_polygon[0]]
        self.color = [[255,0,0],[255,0,255],[0,0,255]]
        self.edges_3d = self.calc_edges_3d()
        self.height = height
        self.num_robots = num_robots

    def start_simulation(self):
        self.trace_line = sim.addDrawingObject(sim.drawing_lines, 5, 0, -1, 9999, [255,0,0]) # red line
        sim.startSimulation()
        print('Program started')

    def stop_simulation(self):
        sim.removeDrawingObject(self.trace_line)
        sim.stopSimulation()

    def calc_edges_3d(self):  # To calculate the edges in the polygon
        edges = []
        for i in range(len(self.rounded_polygon) - 1):
            edges.append([list(self.rounded_polygon[i]), list(self.rounded_polygon[i+1])])
        return edges

    def draw_field(self):
        white = [255, 255, 255]
        lineContainer = sim.addDrawingObject(sim.drawing_lines, 5, 0, -1, 9999, white)
        for l in self.edges_3d: # Drawing the field with white lines
            line = l[0] + [self.height] + l[1] + [self.height]
            for j in range(len(line)):
                if line[j] != self.height:
                    line[j] = int(line[j])
            # print(line)
            sim.addDrawingObjectItem(lineContainer, line)

    def set_agent_positions(self, info):
        for i in range(self.num_robots):
            drone = '/Quadcopter['
            obj_path = drone+str(i)+']'
            objHandle = sim.getObject(obj_path)
            print(np.append(info['robot'+str(i)],[self.height]))
            x = info['robot'+str(i)]
            x = [xi/self.scaling_factor for xi in x]
            x = x + [self.height]
            print(x)
            sim.setObjectPosition(objHandle, -1, x) # Initiate the position of the robots
    
    def set_weed_locations(self, weed_locations):
        weed_obj = sim.getObject('/weed')
        for i, loc in enumerate(weed_locations):
            new_weed_obj = sim.copyPasteObjects([weed_obj])[0]
            x = [xi/self.scaling_factor for xi in loc]
            new_pos = x + [0]
            sim.setObjectPosition(new_weed_obj, -1, new_pos)

    def move_agents(self, info):
        for i in range(self.num_robots):
            obj_path = '/target[' + str(i) + ']'
            objHandle = sim.getObject(obj_path)
            prev_pos = sim.getObjectPosition(objHandle, -1) # current object position
            print(np.append(info['robot'+str(i)],[self.height]))
            x = info['robot'+str(i)] # Get the x,y from info of gym env
            x = [xi/self.scaling_factor for xi in x] # scale the x,y
            x = x + [self.height] # add the z (height)
            # print(x)
            sim.setObjectPosition(objHandle, -1, x) # Initiate the position of the robots
            # draw the line
            line_data = prev_pos + x
            sim.addDrawingObjectItem(self.trace_line, line_data)