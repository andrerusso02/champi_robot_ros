#!/usr/bin/env python3

from icecream import ic
import pygame
from math import atan2, pi, cos, sin
import numpy as np
import pygame_widgets as pw
from pygame_widgets.button import Button
from pygame_widgets.toggle import Toggle

# COLORS
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PINK = (255, 0, 255)

# CONSTANTS
WIDTH, HEIGHT = 900, 600  # window
TABLE_WIDTH, TABLE_HEIGHT = 3, 2  # Table size in m
FPS = 50


class Gui():

    def __init__(self, robot, obstacle, table):

        pygame.init()

        self.robot = robot
        self.obstacle = obstacle
        self.table = table

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Simulateur Robot Holonome")

        self.clock = pygame.time.Clock()
        self.button = Button(
            self.screen, 5, HEIGHT-30-5, 50, 30, text='Pause',
            fontSize=13, margin=20,
            inactiveColour=(100, 100, 100),
            hoverColour=(0, 0, 100),
            radius=20,
            onClick=lambda: self.on_click()
        )

        self.toggle_graph = Toggle(self.screen, 15, HEIGHT-60, 15, 15)
        self.toggle_path = Toggle(self.screen, 15, HEIGHT-85, 15, 15, startOn=True)

        self.waiting_for_release = False
        self.pos_waiting = []

    # Fonction pour convertir les coordonnées réelles en coordonnées écran
    def real_to_screen(self, x, y):
        x = float(x)
        y = float(y)
        x_factor = WIDTH / TABLE_WIDTH
        y_factor = HEIGHT / TABLE_HEIGHT
        # offset to center the table
        offset_x = (WIDTH - TABLE_WIDTH / x_factor)*0.9 / 2
        offset_y = (HEIGHT - TABLE_HEIGHT / y_factor)*0.9 / 2
        # scale everything to 0.9
        screen_x = int(x * x_factor*0.9)+45
        screen_y = int(y * y_factor*0.9)+30

        screen_y = HEIGHT - screen_y # Parce que PyGame et ROS n'ont pas le même sens de l'axe y.

        return screen_x, screen_y

    def draw_roues(self, screen_x, screen_y, robot_theta):
        # Dessiner les roues rectangulaires par 4 lignes
        for i in range(3):
            # 120° d'écart entre les roues

            angles_roues = [pi/2 - 4*pi/3,
                            pi/2 - 2*pi/3,
                            pi/2]
            angle_roue = robot_theta + angles_roues[i]

            # Coordonnées du centre de la roue
            centre_roue = (
                screen_x + self.robot.robot_radius * cos(angle_roue),
                screen_y + self.robot.robot_radius * sin(angle_roue)
            )

            # Coordonnées des coins de la roue
            coin1 = np.array([
                centre_roue[0] + self.robot.wheel_width/2,
                centre_roue[1] + self.robot.wheel_radius
            ])

            coin2 = np.array([
                centre_roue[0] + self.robot.wheel_width/2,
                centre_roue[1] - self.robot.wheel_radius
            ])

            coin3 = np.array([
                centre_roue[0] - self.robot.wheel_width/2,
                centre_roue[1] - self.robot.wheel_radius
            ])

            coin4 = np.array([
                centre_roue[0] - self.robot.wheel_width/2,
                centre_roue[1] + self.robot.wheel_radius
            ])
            # now we rotate each point around the center of the wheel
            coin1 = np.array([
                centre_roue[0] + (coin1[0]-centre_roue[0])*cos(angle_roue) -
                (coin1[1]-centre_roue[1])*sin(angle_roue),
                centre_roue[1] + (coin1[0]-centre_roue[0])*sin(angle_roue) +
                (coin1[1]-centre_roue[1])*cos(angle_roue)
            ])
            coin2 = np.array([
                centre_roue[0] + (coin2[0]-centre_roue[0])*cos(angle_roue) -
                (coin2[1]-centre_roue[1])*sin(angle_roue),
                centre_roue[1] + (coin2[0]-centre_roue[0])*sin(angle_roue) +
                (coin2[1]-centre_roue[1])*cos(angle_roue)
            ])
            coin3 = np.array([
                centre_roue[0] + (coin3[0]-centre_roue[0])*cos(angle_roue) -
                (coin3[1]-centre_roue[1])*sin(angle_roue),
                centre_roue[1] + (coin3[0]-centre_roue[0])*sin(angle_roue) +
                (coin3[1]-centre_roue[1])*cos(angle_roue)
            ])
            coin4 = np.array([
                centre_roue[0] + (coin4[0]-centre_roue[0])*cos(angle_roue) -
                (coin4[1]-centre_roue[1])*sin(angle_roue),
                centre_roue[1] + (coin4[0]-centre_roue[0])*sin(angle_roue) +
                (coin4[1]-centre_roue[1])*cos(angle_roue)
            ])

            # Dessiner le polygone représentant la roue
            pygame.draw.polygon(self.screen, BLACK, [coin1, coin2, coin3, coin4])

            # axes des vecteurs vitesses unitaires de la roue en rouge
            v_x = ((coin2 + coin3)/2 - centre_roue)
            v_x = v_x / np.linalg.norm(v_x)
            v_y = ((coin2 + coin1)/2 - centre_roue)
            v_y = v_y / np.linalg.norm(v_y)
            pygame.draw.line(self.screen, BLACK, centre_roue, centre_roue + v_x * 20, 2)
            pygame.draw.line(self.screen, BLACK, centre_roue, centre_roue + v_y * 20, 2)

            # texte numero de roue
            font = pygame.font.SysFont('Arial', 10)
            text = font.render(str(i), True, BLACK)
            textRect = text.get_rect()
            textRect.center = (v_y[0]*10+v_x[0]*10+centre_roue[0],
                            v_y[1]*10+v_x[1]*10+centre_roue[1])
            self.screen.blit(text, textRect)

    def draw_poly(self, obstacle, color, width=1):
        vertices = list(obstacle.exterior.coords)
        vertices.append(vertices[0])
        for i in range(len(vertices)-1):
            A = self.real_to_screen(float(vertices[i][0]),float(vertices[i][1]))
            B = self.real_to_screen(float(vertices[i+1][0]),float(vertices[i+1][1]))
            pygame.draw.line(self.screen, color, A, B, width)

    def draw_graph_and_path(self, dico_all_points):
        if self.toggle_graph.getValue()==True:
            # draw the graph
            for A,B in self.robot.graph.items():
                pointA = dico_all_points[A]
                for b in B.keys():
                    pointB = dico_all_points[b]
                    pygame.draw.line(self.screen, BLACK, 
                                    self.real_to_screen(float(pointA[0]),float(pointA[1])), 
                                    self.real_to_screen(float(pointB[0]),float(pointB[1])),
                                    3)
                    
                    # font = pygame.font.SysFont('Arial', 20)
                    # text = font.render(str(A), True, RED)
                    # textRect = text.get_rect()
                    # textRect.center = real_to_screen(float(pointA[0]),float(pointA[1]))
                    # screen.blit(text, textRect)
                
        if self.toggle_path.getValue()==True:
            # draw path
            if self.robot.path_nodes is not None:
                nodes = self.robot.path_nodes
                for i in range(len(nodes)-1):
                    pygame.draw.line(self.screen, GREEN, 
                        self.real_to_screen(dico_all_points[nodes[i]][0],dico_all_points[nodes[i]][1]), 
                        self.real_to_screen(dico_all_points[nodes[i+1]][0], dico_all_points[nodes[i+1]][1]), 
                        5)

        # draw the obstacle
        self.draw_poly(self.obstacle.polygon, RED)    
        self.draw_poly(self.obstacle.expanded_obstacle_poly, YELLOW)  

    def draw(self):
        # drawings
        self.screen.fill(WHITE)

        self.draw_poly(self.table.polygon, BLACK, 3)
        self.draw_poly(self.table.expanded_poly, YELLOW)

        # Dessiner les positions du robot
        for pos in self.robot.robot_positions: 
            screen_x, screen_y = self.real_to_screen(pos[0], pos[1])
            pygame.draw.circle(self.screen, BLUE, (screen_x, screen_y), 1)

        # draw all next goals by a cross and an arrow
        for i in range(len(self.robot.goals_positions)):
            screen_x, screen_y = self.real_to_screen(
                self.robot.goals_positions[i][0], self.robot.goals_positions[i][1])
            pygame.draw.line(self.screen, GREEN, (screen_x-5, screen_y-5),
                            (screen_x+5, screen_y+5), 2)
            pygame.draw.line(self.screen, GREEN, (screen_x+5, screen_y-5),
                            (screen_x-5, screen_y+5), 2)
            # arrow indicating the angle
            pygame.draw.line(self.screen, GREEN, (screen_x, screen_y),
                            (screen_x+20*cos(self.robot.goals_positions[i][2]-pi/2),
                                screen_y+20*sin(self.robot.goals_positions[i][2]-pi/2)), 2)
            # text
            angle = self.robot.goals_positions[i][2]*180/pi
            font = pygame.font.SysFont('Arial', 10)
            text = font.render(str(angle)+"°", True, BLACK)
            textRect = text.get_rect()
            textRect.center = (screen_x, screen_y)
            self.screen.blit(text, textRect)

        # Dessiner le robot à sa nouvelle position
        screen_x, screen_y = self.real_to_screen(self.robot.pos[0], self.robot.pos[1])
        pygame.draw.circle(self.screen, BLACK, (screen_x, screen_y),
                        self.robot.robot_radius, 1)
        pygame.draw.circle(self.screen, GREEN, (screen_x, screen_y), 1)

        # dessiner la direction du robot
        robot_angle_gui = - self.robot.pos[2] # Parce que PyGame et ROS ne tournent pas dans le même sens.
        pygame.draw.line(self.screen, GREEN, (screen_x, screen_y),
                        (screen_x+20*cos(robot_angle_gui), screen_y+20*sin(robot_angle_gui)), 2)

        self.draw_roues(screen_x, screen_y, robot_angle_gui)

        self.draw_graph_and_path(self.robot.dico_all_points)

        # print the speeds in the top left corner
        font = pygame.font.SysFont('Arial', 20)
        text = font.render(
            "speed_x: " + str(round(self.robot.linear_speed[0], 2)) + " m/s", True, BLACK)
        textRect = text.get_rect()
        textRect.topleft = (50, 30)
        self.screen.blit(text, textRect)
        text = font.render(
            "speed_y: " + str(round(self.robot.linear_speed[1], 2)) + " m/s", True, BLACK)
        textRect = text.get_rect()
        textRect.topleft = (50, 50)
        self.screen.blit(text, textRect)
        text = font.render(
            "speed_theta: " + str(round(self.robot.angular_speed, 2)) + " rad/s", True, BLACK)
        textRect = text.get_rect()
        textRect.topleft = (50, 70)
        self.screen.blit(text, textRect)
        text = font.render(
            "theta: " + str(round(self.robot.pos[2], 2)) + " rad", True, BLACK)
        textRect = text.get_rect()
        textRect.topleft = (50, 90)
        self.screen.blit(text, textRect)

        self.button.draw()
        self.toggle_graph.draw()
        self.toggle_path.draw()

        if self.waiting_for_release:
            # draw a line from pos_waiting to the mouse
            mouse_pos = pygame.mouse.get_pos()
            x_screen, y_screen = mouse_pos
            x, y = x_screen-45, HEIGHT - y_screen-30
            x, y = x/(WIDTH-90)*TABLE_WIDTH, y/(HEIGHT-60)*TABLE_HEIGHT
            pygame.draw.line(self.screen, BLACK, self.real_to_screen(
                self.pos_waiting[0], self.pos_waiting[1]), self.real_to_screen(x, y), 2)
            
            theta = atan2(y-self.pos_waiting[1], x-self.pos_waiting[0])
            # draw text to show the angle
            font = pygame.font.SysFont('Arial', 10)
            text = font.render(str(round(theta*180/pi, 2))+"°", True, RED)
            textRect = text.get_rect()
            textRect.center = (x_screen, y_screen)
            self.screen.blit(text, textRect)

        # Mettre à jour l'affichage
        pygame.display.flip() 

    def on_click(self):
        global robot
        print("button clicked")
        self.robot.linear_speed = [0, 0]


    def update(self):
        # EVENTS
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    quit()

            # si un clic souris, on va à la position du clic
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: # left click down
                mouse_pos = pygame.mouse.get_pos()
                x, y = mouse_pos
                x, y = x-45, HEIGHT - y-30
                x, y = x/(WIDTH-90)*TABLE_WIDTH, y/(HEIGHT-60)*TABLE_HEIGHT
                self.waiting_for_release = True
                self.pos_waiting = [x, y]

            if event.type == pygame.MOUSEBUTTONUP and self.waiting_for_release and event.button == 1: # left click up
                # calcule la position sur la table
                mouse_pos = pygame.mouse.get_pos()
                x, y = mouse_pos
                x, y = x-45, HEIGHT - y-30
                x, y = x/(WIDTH-90)*TABLE_WIDTH, y/(HEIGHT-60)*TABLE_HEIGHT
                # calcule l'angle demandé
                theta = atan2(y-self.pos_waiting[1], x-self.pos_waiting[0])
                
                # GOTO
                ic("goto", self.pos_waiting, theta*180/pi)
                self.waiting_for_release = False

                """pour le node pose_control TODO EST CE QUE C'EST BIEN COMMME CA?"""
                # self.robot.goals_positions.append([self.pos_waiting[0], self.pos_waiting[1], theta]) # TODO remettre

            # si clic droit on bouge le robot adverse
            if pygame.mouse.get_pressed()[2]:
                if event.type == pygame.MOUSEMOTION or event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.MOUSEBUTTONUP:
                    # if event.button == 3: # right click
                    mouse_pos = pygame.mouse.get_pos()
                    x, y = mouse_pos
                    x, y = x-45, HEIGHT - y-30
                    x, y = x/(WIDTH-90)*TABLE_WIDTH, y/(HEIGHT-60)*TABLE_HEIGHT

                    """ENVOI MESSAGE ROS"""
                    # obstacle = Obstacle_static_model(x, y, 10, 10, offset) TODO

        self.button.listen(events)
        pw.update(events)

        # delete the last point so that the array does not get to heavy
        if len(self.robot.robot_positions)>500:
            self.robot.robot_positions.pop(0)

        """ENVOI MSG ROS POSE GOAL, POLY ROBOT ADVERSE"""

        self.draw()    
        self.clock.tick(FPS)
