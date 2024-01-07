import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
from class_utils import Room, House
import argparse

def reader(filename):
    with open(filename) as f:
        info =json.load(f)
        rms_bbs=np.asarray(info['boxes'])
        fp_eds=info['edges']
        rms_type=info['room_type']
        eds_to_rms=info['ed_rm']

        edges = np.array(fp_eds)
        edges = edges[:,:4]

        s_r=0
        for rmk in range(len(rms_type)):
            if(rms_type[rmk]!=17):
                s_r=s_r+1   
        rms_bbs = np.array(rms_bbs)/256.0
        fp_eds = np.array(fp_eds)/256.0 
        fp_eds = fp_eds[:, :4]
        tl = np.min(rms_bbs[:, :2], 0)
        br = np.max(rms_bbs[:, 2:], 0)
        shift = (tl+br)/2.0 - 0.5
        rms_bbs[:, :2] -= shift 
        rms_bbs[:, 2:] -= shift
        fp_eds[:, :2] -= shift
        fp_eds[:, 2:] -= shift 
        return rms_type,fp_eds,rms_bbs,eds_to_rms

def make_sequence(edges):
    polys = []
    v_curr = tuple(edges[0][:2])
    e_ind_curr = 0
    e_visited = [0]
    seq_tracker = [v_curr]
    find_next = False
    while len(e_visited) < len(edges):
        if find_next == False:
            if v_curr == tuple(edges[e_ind_curr][2:]):
                v_curr = tuple(edges[e_ind_curr][:2])
            else:
                v_curr = tuple(edges[e_ind_curr][2:])
            find_next = not find_next 
        else:
            # look for next edge
            for k, e in enumerate(edges):
                if k not in e_visited:
                    if (v_curr == tuple(e[:2])):
                        v_curr = tuple(e[2:])
                        e_ind_curr = k
                        e_visited.append(k)
                        break
                    elif (v_curr == tuple(e[2:])):
                        v_curr = tuple(e[:2])
                        e_ind_curr = k
                        e_visited.append(k)
                        break

        # extract next sequence
        if v_curr == seq_tracker[-1]:
            polys.append(seq_tracker)
            for k, e in enumerate(edges):
                if k not in e_visited:
                    v_curr = tuple(edges[0][:2])
                    seq_tracker = [v_curr]
                    find_next = False
                    e_ind_curr = k
                    e_visited.append(k)
                    break
        else:
            seq_tracker.append(v_curr)
    polys.append(seq_tracker)

    return polys

def build_graph(rms_type, fp_eds, eds_to_rms, out_size=64):
    # create edges
    triples = []
    nodes = rms_type 
    # encode connections
    for k in range(len(nodes)):
        for l in range(len(nodes)):
            if l > k:
                is_adjacent = any([True for e_map in eds_to_rms if (l in e_map) and (k in e_map)])
                if is_adjacent:
                    triples.append([k, 1, l])
                else:
                    triples.append([k, -1, l])
    # get rooms masks
    eds_to_rms_tmp = []
    for l in range(len(eds_to_rms)):                  
        eds_to_rms_tmp.append([eds_to_rms[l][0]])
    rms_masks = []
    im_size = 256
    fp_mk = np.zeros((out_size, out_size))
    for k in range(len(nodes)):
        # add rooms and doors
        eds = []
        for l, e_map in enumerate(eds_to_rms_tmp):
            if (k in e_map):
                eds.append(l)
        # draw rooms
        rm_im = Image.new('L', (im_size, im_size))
        dr = ImageDraw.Draw(rm_im)
        for eds_poly in [eds]:
            poly = make_sequence(np.array([fp_eds[l][:4] for l in eds_poly]))[0]
            poly = [(im_size*x, im_size*y) for x, y in poly]
            if len(poly) >= 2:
                dr.polygon(poly, fill='white')
            else:
                print("Empty room")
                exit(0)
        rm_im = rm_im.resize((out_size, out_size))
        rm_arr = np.array(rm_im)
        inds = np.where(rm_arr>0)
        rm_arr[inds] = 1.0
        rms_masks.append(rm_arr)
        if rms_type[k] != 15 and rms_type[k] != 17:
            fp_mk[inds] = k+1
    # trick to remove overlap
    for k in range(len(nodes)):
        if rms_type[k] != 15 and rms_type[k] != 17:
            rm_arr = np.zeros((out_size, out_size))
            inds = np.where(fp_mk==k+1)
            rm_arr[inds] = 1.0
            rms_masks[k] = rm_arr
    # convert to array
    nodes = np.array(nodes)
    triples = np.array(triples)
    rms_masks = np.array(rms_masks)
    return nodes, triples, rms_masks

def create_house_from_name(file_name, reso):
    rms_type, fp_eds,rms_bbs,eds_to_rms=reader(file_name) 
    a = [rms_type, rms_bbs, fp_eds, eds_to_rms]
    
    rms_bbs = np.array(rms_bbs)
    fp_eds = np.array(fp_eds)

    # extract boundary box and centralize
    tl = np.min(rms_bbs[:, :2], 0)
    br = np.max(rms_bbs[:, 2:], 0)
    shift = (tl+br)/2.0 - 0.5
    rms_bbs[:, :2] -= shift
    rms_bbs[:, 2:] -= shift
    fp_eds[:, :2] -= shift
    fp_eds[:, 2:] -= shift
    tl -= shift
    br -= shift

    # build input graph
    graph_nodes, graph_edges, rooms_mks = build_graph(rms_type, fp_eds, eds_to_rms)

    house = []
    for room_mask, room_type in zip(rooms_mks, graph_nodes):
        room_mask = room_mask.astype(np.uint8)
        room_mask = cv2.resize(room_mask, (256, 256), interpolation = cv2.INTER_AREA)
        contours, _ = cv2.findContours(room_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0]
        contours = contours / 256 * reso
        house.append([contours[:,0,:], room_type])

    graph = graph_edges
    ### house: contain edges and type for each room
    ### graph: contain connection

    room_classes = []
    for room_idx in range(len(house)):
        corners = house[room_idx][0]
        room_type = house[room_idx][1]
        room = Room(corners, room_type, corner_threshold=reso/64)
        room_classes.append(room)
    
    # add connection
    for graph_i in range(graph.shape[0]):
        connection = graph[graph_i]
        room_idx0 = connection[0]
        room_idx1 = connection[2]
        room0 = room_classes[room_idx0]
        room1 = room_classes[room_idx1]
        is_adjacent = connection[1]
        if is_adjacent == 1:
            room0.add_neighbor(room1)
            room1.add_neighbor(room0)
    
    house = House(room_classes)
    
    return house

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', default=1, type=int)
    args = parser.parse_args()

    base_dir = 'datasets/rplan'
    target_set = -1
    reso = 64


    subgraphs = []
    subnames = []
    max_room_num = 16
    print(max_room_num)
    with open(f'{base_dir}/list.txt') as f:
        lines = f.readlines()
        cnt=0

        m = len(lines) // 25
        for line in tqdm(lines[m*args.group:m*(args.group+1)]):
        #for line in tqdm(lines):
            cnt=cnt+1
            file_name = f'{base_dir}/{line[:-1]}'
            rms_type, fp_eds,rms_bbs,eds_to_rms=reader(file_name) 
            fp_size = len([x for x in rms_type if x != 15 and x != 17])
            if fp_size == target_set:
                    continue
            a = [rms_type, rms_bbs, fp_eds, eds_to_rms]

            rms_bbs = np.array(rms_bbs)
            fp_eds = np.array(fp_eds)

            # extract boundary box and centralize
            tl = np.min(rms_bbs[:, :2], 0)
            br = np.max(rms_bbs[:, 2:], 0)
            shift = (tl+br)/2.0 - 0.5
            rms_bbs[:, :2] -= shift
            rms_bbs[:, 2:] -= shift
            fp_eds[:, :2] -= shift
            fp_eds[:, 2:] -= shift
            tl -= shift
            br -= shift

            # build input graph
            graph_nodes, graph_edges, rooms_mks = build_graph(rms_type, fp_eds, eds_to_rms)

            house = []
            for room_mask, room_type in zip(rooms_mks, graph_nodes):
                room_mask = room_mask.astype(np.uint8)
                room_mask = cv2.resize(room_mask, (256, 256), interpolation = cv2.INTER_AREA)
                contours, _ = cv2.findContours(room_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[0]
                contours = contours / 256 * reso
                house.append([contours[:,0,:], room_type])

            graph = graph_edges
            name = line.split('.')[0]

            ### house: contain edges and type for each room
            ### graph: contain connection

            room_classes = []
            if len(house) > max_room_num:
                continue
            for room_idx in range(len(house)):
                corners = house[room_idx][0]
                room_type = house[room_idx][1]
                room = Room(corners, room_type, corner_threshold=reso/64)
                room_classes.append(room)

            # add connection
            for graph_i in range(graph.shape[0]):
                connection = graph[graph_i]
                room_idx0 = connection[0]
                room_idx1 = connection[2]
                room0 = room_classes[room_idx0]
                room1 = room_classes[room_idx1]
                is_adjacent = connection[1]
                if is_adjacent == 1:
                    room0.add_neighbor(room1)
                    room1.add_neighbor(room0)

            house = House(room_classes)

            rep, meta = house.prepare_data(max_room_num=max_room_num, reso=reso)
            data = {
                'x': rep,
                'meta': meta,
                'filename': name
            }

            # save
            np.save(f'data_bank/train_{name}', data)


                



        