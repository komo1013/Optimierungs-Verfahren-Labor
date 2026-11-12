import manim
import manim.opengl
import numpy as np
from manim import PI
from manim import PINK, BLUE, GREEN, RED
from manim import RIGHT, LEFT, UP, DOWN, OUT, IN


'''References'''

class CreateCircle(manim.Scene):
    def construct(self):
        circle = manim.Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set the color and transparency
        self.play(manim.Create(circle))  # show the circle on screen

class SquareToCircle(manim.Scene):
    def construct(self):
        circle = manim.Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set color and transparency

        square = manim.Square()  # create a square
        square.rotate(PI / 4)  # rotate a certain amount

        self.play(manim.Create(square))  # animate the creation of the square
        self.play(manim.Transform(square, circle))  # interpolate the square into the circle
        self.play(manim.FadeOut(square))  # fade out animation

class SquareAndCircle(manim.Scene):
    def construct(self):
        circle = manim.Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set the color and transparency

        square = manim.Square()  # create a square
        square.set_fill(BLUE, opacity=0.5)  # set the color and transparency

        square.next_to(circle, RIGHT, buff=0.5)  # set the position
        self.play(manim.Create(circle), manim.Create(square))  # show the shapes on screen

class AnimatedSquareToCircle(manim.Scene):
    def construct(self):
        circle = manim.Circle()  # create a circle
        square = manim.Aquade()  # create a square

        self.play(manim.Create(square))  # show the square on screen
        self.play(square.animate.rotate(PI / 4))  # rotate the square
        self.play(manim.Transform(square, circle))  # transform the square into a circle
        self.play(
            square.animate.set_fill(PINK, opacity=0.5)
        )  # color the circle on screen

class DifferentRotations(manim.Scene):
    def construct(self):
        left_square = manim.Square(color=BLUE, fill_opacity=0.7).shift(2 * LEFT)
        right_square = manim.Square(color=GREEN, fill_opacity=0.7).shift(2 * RIGHT)
        self.play(
            left_square.animate.rotate(PI), manim.Rotate(right_square, angle=PI), run_time=2
        )
        self.wait()

class TwoTransforms(manim.Scene):
    def transform(self):
        a = manim.Circle()
        b = manim.Square()
        c = manim.Triangle()
        self.play(manim.Transform(a, b))
        self.play(manim.Transform(a, c))
        self.play(manim.FadeOut(a))

    def replacement_transform(self):
        a = manim.Circle()
        b = manim.Square()
        c = manim.Triangle()
        self.play(manim.ReplacementTransform(a, b))
        self.play(manim.ReplacementTransform(b, c))
        self.play(manim.FadeOut(c))

    def construct(self):
        self.transform()
        self.wait(0.5)  # wait for 0.5 seconds
        self.replacement_transform()

class TestCamRot(manim.ThreeDScene):
    def construct(self):
        self.set_camera_orientation(PI / 3, -PI / 4)
        ax = manim.ThreeDAxes((-5, 5), (-5, 5), (-5, 5), 10, 10, 10)
        line1 = manim.Line3D(RIGHT * 2, UP + OUT, color=RED)
        line2 = manim.Line3D.perpendicular_to(line1, color=BLUE)
        self.add(ax, line1, line2)
        self.begin_ambient_camera_rotation(rate=0.4)
        self.wait(5)

class LinearSpline(manim.Scene):
    def place_points(self):
        coordinates = np.array([[0,0,0],[1,0,0],[1.5,0.5,0],[2,0.5,0],[1,2,0],[1.5, 2.7,0],[-0.5, 2.5,0], [-0.7, 0.5,0],[0, 1.5, 0],[1, 0.7,0]])
        points = np.array([])
        lines = np.array([])
        for i, location in enumerate(coordinates):
            point = manim.Circle(radius=0.05, color=BLUE, fill_opacity=0.8).move_to(location)
            #points = np.append(points, point)
            if (i != 0):
                line = manim.Line(coordinates[i-1], coordinates[i] ,color=GREEN, buff=0.05)
                #lines = np.append(lines, line)
                self.add(line)
                #self.play(manim.FadeIn(line, scale=1.2))
            self.add(point)
            #self.play(manim.FadeIn(point, scale=1.2))
            #self.wait(0.05)
        line = manim.Line(coordinates[-1], coordinates[0], color=GREEN, buff=0.05)
        self.add(line)
    def construct(self):
        self.place_points()
