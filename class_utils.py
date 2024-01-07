import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def filter_close_points(points, min_distance=4):
    filtered_points = []
    i = 0
    for i in range(points.shape[0]):
        if i == 0:
            filtered_points.append(points[i])
        else:
            distance = np.linalg.norm(points[i]-filtered_points[-1])
            if distance > min_distance:
                filtered_points.append(points[i])

    return np.array(filtered_points)


class House():
    def __init__(self, rooms):
        # lexicographical order
        self.rooms = sorted(rooms, key=lambda room: list(room.corners[0]))

    def prepare_data(self, max_room_num=16, reso=64):
        assert len(self.rooms) <= max_room_num
        rep = np.zeros((max_room_num, reso, reso))
        meta = {'room_num': len(self.rooms),
                'inside_room_index': np.ones((reso,reso))*-1}
        for i in range(rep.shape[-2]):
            for j in range(rep.shape[-1]):
                point = np.array([i,j])
                dists = [room.point_distance(point) for room in self.rooms]
                rep[:len(dists),i,j] = np.array(dists)
        
        if False:
            fig, ax = plt.subplots()
            ax.set_xlim(0,reso-1)
            ax.set_ylim(reso-1,0)
            norm = mcolors.TwoSlopeNorm(vmin=-32, vcenter=0, vmax=10)
            ax.imshow(rep[0], cmap='RdBu', norm=norm)
            self.render(ax, render_mid_points=False,fig_path=None, reso=64)
            plt.savefig('signed.png')
            fig, ax = plt.subplots()
            ax.set_xlim(0,reso-1)
            ax.set_ylim(reso-1,0)
            norm = mcolors.TwoSlopeNorm(vmin=-32, vcenter=0, vmax=10)
            ax.imshow(np.abs(rep[0]), cmap='RdBu', norm=norm)
            self.render(ax, render_mid_points=False,fig_path=None, reso=64)
            plt.savefig('unsigned.png')
        return rep, meta

            
    def render(self, ax=None, render_mid_points=False, fig_path=None, reso=64):
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_xlim(0,reso-1)
            ax.set_ylim(reso-1,0)
            ax.imshow(np.ones((reso,reso,3)))
        for room in self.rooms:
            room.render(ax, render_mid_points, None, False, reso=reso)
        
        if fig_path is not None:
            plt.savefig(fig_path, dpi=150)
            plt.close()

class Room():
    def __init__(self, corners, room_type, corner_threshold=4):
        self.room_type = room_type #15, 17 are doors
        
        ## sometimes two neighbor corners are too close
        corners = filter_close_points(corners, min_distance=corner_threshold)

        # lexicographical order corners
        # corners: Nx2
        smallest_index = np.lexsort((corners[:, 1], corners[:, 0]))[0]
        self.corners = np.roll(corners, -smallest_index, axis=0)

        # clockwise order
        if self.corners[-1,0] < self.corners[1,0]:
            order = np.roll(np.arange(self.corners.shape[0])[::-1], 1) # 0, 3, 2, 1
            self.corners = self.corners[order]

        self.make_walls()
        self.neighbors = []
    
    def add_neighbor(self, room):
        assert isinstance(room, Room)
        if self.neighbors is None:
            self.neighbors = []
        self.neighbors.append(room)
        # lexicographical order
        self.neighbors = sorted(self.neighbors, key=lambda room: list(room.corners[0]))

    def make_walls(self):
        self.walls = []
        for corner_i in range(self.corners.shape[0]):
            corner_j = (corner_i+1)%self.corners.shape[0]
            self.walls.append(Wall(self.corners[corner_i], self.corners[corner_j]))

    def get_neighbors(self):
        return self.neighbors

    def is_neighbor(self, room):
        if room in self.neighbors:
            return True
        return False

    def point_distance(self, point):
        min_distance = 512
        for wall in self.walls:
            current_distance = wall.point_distance(point)
            if current_distance < min_distance:
                min_distance = current_distance
        
        if self.is_point_inside(point):
            return min_distance
        else:
            return -min_distance

    def is_point_inside(self, point):
        x, y = point
        inside = False

        for i in range(self.corners.shape[0]):
            x1, y1 = self.corners[i]
            x2, y2 = self.corners[(i + 1) % self.corners.shape[0]]

            if min(y1, y2) < y <= max(y1, y2) and x <= max(x1, x2):
                if y1 != y2:
                    xinters = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                if x1 == x2 or x <= xinters:
                    inside = not inside
        return inside

    def is_door(self):
        if self.room_type == 15 or self.room_type == 17:
            return True
        return False

    def render(self, ax=None, render_mid_points=False, fig_path=None, 
                render_neighbors=False, is_neighbor=False, reso=64):
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_xlim(0,reso)
            ax.set_ylim(0,reso)
        for wall in self.walls:
            wall.render(ax, render_mid_points, fig_path=None, 
                        dash_line=is_neighbor, reso=reso)
        
        if render_neighbors:
            for neighbor in self.neighbors:
                neighbor.render(ax, render_mid_points, None, False, 
                                is_neighbor=True, reso=reso)
        
        if fig_path is not None:
            plt.savefig(fig_path, dpi=150)
            plt.close()
        

class Wall():
    def __init__(self, start, end, num_mid_points=5):
        self.start = start
        self.end = end

        # interpolate
        scales = np.linspace(0,1,num_mid_points+2)[1:-1][:,None] # m,1
        self.mid_points = (1-scales) * start[None,:] + scales * end[None,:] # m,2

    def point_distance(self, point):
        p = np.array(point)
        a = np.array(self.start)
        b = np.array(self.end)

        ab = b - a
        ap = p - a

        # Projecting vector ap onto vector ab
        t = np.dot(ap, ab) / np.dot(ab, ab)
        if t < 0.0:
            # Point projection is before a, so use distance to a
            closest = a
        elif t > 1.0:
            # Point projection is after b, so use distance to b
            closest = b
        else:
            # Projection falls on the line segment, calculate the exact point
            closest = a + t * ab

        return np.linalg.norm(p - closest)

    def get_represented_points(self):
        p = np.concatenate((self.start[None], self.mid_points), 0)
        return p

    def render(self, ax=None, render_mid_points=False, 
               fig_path=None, dash_line=False, reso=64):
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_xlim(0,reso)
            ax.set_ylim(0,reso)
        line_style = '--' if dash_line else '-' 
        ax.plot([self.start[1], self.end[1]], [self.start[0], self.end[0]], line_style)
        if render_mid_points:
            ax.scatter(self.mid_points[:,1], self.mid_points[:,0],s=2)
        if fig_path is not None:
            plt.savefig(fig_path, dpi=150)
            plt.close()
        

