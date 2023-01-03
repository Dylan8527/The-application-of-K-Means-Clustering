import polyscope as ps
import numpy as np
from load_obj import *
import random
import time
import itertools
import sys
import polyscope.imgui as psim
import json
from queue import PriorityQueue
from functools import total_ordering
import argparse

def calculateAreaOfTriangularFace(vect1, vect2):
    # cross product is double area of triangle -> 0.5 * cross product = area of triangle
    return np.linalg.norm(
        np.array(
            [
                vect1[1] * vect2[2] - vect1[2] * vect2[1],
                vect1[2] * vect2[0] - vect1[0] * vect2[2],
                vect1[0] * vect2[1] - vect1[1] * vect2[0]
            ]
        )
    ) * .5


class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces # `faces` is a list of triples of vertex indices (i.e. a list of triangles)
        self.edges = None

    def getAllFacesArea(self):
        areas = []
        for face in self.faces:
            areas.append(
                calculateAreaOfTriangularFace(
                    self.vertices[face[1]] - self.vertices[face[0]],
                    self.vertices[face[2]] - self.vertices[face[0]]
                )
            )
        return areas

    def getAllFacesNormals(self):
        normals = []
        for face in self.faces:
            U = self.vertices[face[1]] - self.vertices[face[0]]
            V = self.vertices[face[2]] - self.vertices[face[0]]
            normals.append(
                (
                    U[1] * V[2] - U[2] * V[1],
                    U[2] * V[0] - U[0] * V[2],
                    U[0] * V[1] - U[1] * V[0]
                )
            )
        return normals

    def getAllAdjacentFaces(self):
        if self.edges is None:
            raise AssertionError("You have to run getAllFaceEdges before calling getAllAdjacentFaces")
        ajdF = [set() for i in range(len(self.faces))] 
        #! edge(i, j) exists iff adjF[i] contains j and adjF[j] contains i
        corrEdges = {}
        for i in range(len(self.edges)):
            for j in self.edges[i]:
                if corrEdges.get(j,None) is not None:
                    ajdF[corrEdges[j]].add(i)
                    ajdF[i].add(corrEdges[j])
                else:
                    corrEdges[j] = i
        return ajdF

    def getAllFaceEdges(self):
        faceEdges = []
        correspondance = {}
        for face in self.faces:
            fE = [] # store three edges for each face
            for i in range(3):
                key = coordsToString(self.vertices[face[i]], self.vertices[face[i-1]])
                idE = correspondance.get(key, None)
                if idE is None:
                    correspondance[key] = len(correspondance.keys()) # index the edge
                    idE = correspondance[key]
                fE.append(idE)
            faceEdges.append(fE) # actually also a triangle
        self.edges = faceEdges
        return faceEdges

def coStr(nb):
    return str(nb) if nb < 0 else "+" + str(nb)

def coordsToString(c1, c2):
    if ordonne(c1, c2):
        tmp = c1
        c1 = c2
        c2 = tmp
    return coStr(c1[0]) + coStr(c2[0]) + coStr(c1[1]) + coStr(c2[1]) + coStr(c1[2]) + coStr(c2[2])

def ordonne(vectice1, vectice2):
    if vectice1[0] != vectice2[0]:
        return vectice1[0] > vectice2[0]
    if vectice1[1] != vectice2[1]:
        return vectice1[1] > vectice2[1]
    return vectice1[2] > vectice2[2]

class Proxy:
    def __init__(self, regionIndex, faceIndexes, proxyNormal=None):
        self.regionIndex = regionIndex # unique index for each region
        self.faceIndexes = faceIndexes # face indices belonging to this region
        self.proxyNormal = proxyNormal # area_weighted normal
        self.polyMesh = None

    def draw(self, color, globalVertices, globalFaces):
        if self.polyMesh:
            self.undraw()
        self.polyMesh = generateId()
        verticesProxy, facesProxy = GrowSeeds(self.faceIndexes, globalFaces, globalVertices)
        ps.register_surface_mesh(self.polyMesh, verticesProxy, facesProxy, color=color)

    def undraw(self):
        try:
            if self.polyMesh:
                ps.remove_surface_mesh(self.polyMesh)
            else:
                raise TypeError
        except TypeError:
            pass

    def __str__(self):
        return self.regionIndex + " : " + str(self.faceIndexes) + " - " + str(self.proxyNormal) + " - " + str(
            self.polyMesh)

@total_ordering
class QueueElement:
    def __init__(self, error, regionIndex, index): # store the L2 distance between a region with a face
        self.error = error
        self.regionIndex = regionIndex
        self.index = index

    def __str__(self):
        return "Face " + str(self.index) + " of region " + str(self.regionIndex) + " : " + str(self.error)

    def __gt__(self, other): # for priority queue
        return self.error > other.error

    def __eq__(self, other): # for priority queue
        return self.error == other.error

"""
Main algorithm: K-Means clustering
"""
def KMeans(n, proxys, faceNormals, vertices, faceVertexIndexes, areaFaces, faceEdges, adjacentToFaces):
    # Initial stage: proxys
    # Each iteration: 1. calcualte regions' area_weighted normals
    #                 2. calcualte L2-distance between all faces and all regions w.r.t normals -> proxys
    #                 3. put udpated proxys into priority queue
    #                 4. repeat poping the top element from the queue until all faces are assigned to a region
    #                 5. split the worst region into two regions -> increase one region
    #                 6. update adjacentRegions information 
    #                 7. merge two best regions into two region -> decrease one region
    #                 8. repeat 1-7  n times
    for i in range(n):
        # 1. calcualte regions' area_weighted normals
        t=time.time()
        proxys = GetProxy(proxys)
        print("GetProxy :", time.time() - t)
        # 2. calcualte L2-distance between all faces and all regions w.r.t normals -> proxys
        t=time.time()
        regions = GetProxySeed(proxys, faceNormals, areaFaces)
        print("GetProxySeed :", time.time() - t)
        # 3. put udpated proxys into priPyramideority queue
        t=time.time()
        queue, assignedIndexes = BuildQueue(regions, faceNormals, areaFaces, adjacentToFaces)
        print("BuildQueue :", time.time() - t)
        # 4. repeat poping the top element from the queue until all faces are assigned to a region
        t=time.time()
        regions, worst = AssignToRegion(faceNormals, areaFaces, adjacentToFaces, regions, queue, assignedIndexes)
        print("AssignToRegion :", time.time() - t)
        # 5. split the worst region into two regions -> increase one region
        t=time.time()
        regions = SplitRegion(faceNormals, areaFaces, adjacentToFaces, regions, worst)
        print("SplitRegion :", time.time() - t)
        # 6. update adjacentRegions information 
        t=time.time()
        adjacentRegions = FindAdjacentRegions(faceEdges, regions)
        print("FindAdjacentRegions :", time.time() - t)
        # 7. merge two best regions into two region -> decrease one region
        t=time.time()
        regions = FindRegionsToCombine(regions, adjacentRegions, faceNormals, areaFaces)
        print("FindRegionsToCombine :", time.time() - t)
        proxys = regions
    return proxys

"""
refresh color for polyscope
"""
def RefreshAllProxys(oldProxys, newProxys, mesh):
    for proxy in oldProxys.values():
        proxy.undraw()
    for proxy in newProxys.values():
        proxy.draw(Randomcolor(), mesh.vertices, mesh.faces)


"""
return the index th region
"""
def FindRegion(regions, index):
    return regions.get(index)


"""
delete 'regionIndex' region and return regions
"""
def RemoveRegion(regions, regionIndex):
    del regions[regionIndex]
    return regions


"""
Given a region and a list of faces, we generate face-region pairs proxys 
"""
def calculateNewElementsOfQueue(queue, regionIndex, faces, proxyNormal, areaFaces, faceNormals,
                                isInFindRegionToCombine=False, paramsFindRegionToCombine=None):
    for index in faces:
        area = areaFaces[index]
        normal = faceNormals[index]
        try:
            normalError = normal - proxyNormal
        except TypeError:
            proxyNormal = np.array([0, 0, 0])
            normalError = normal - proxyNormal
        moduleNormalError = (np.linalg.norm(normalError)) ** 2 # L2 distance w.r.t normals
        error = moduleNormalError * area

        if isInFindRegionToCombine:
            if error > paramsFindRegionToCombine["maxError"] and paramsFindRegionToCombine["i"] > 0:
                break
            else:
                paramsFindRegionToCombine["regionsToCombine"] = paramsFindRegionToCombine["mergedRegion"]
                paramsFindRegionToCombine["maxError"] = error
        else:
            queue.put(QueueElement(error, regionIndex, index))
    return paramsFindRegionToCombine if isInFindRegionToCombine else queue


def MetricError(regionIndex, faceIndexes, faceNormals, areaFaces, proxyNormal):
    return calculateNewElementsOfQueue(PriorityQueue(), regionIndex, faceIndexes, proxyNormal, areaFaces, faceNormals)
def UpdateQueueNew(region, faceNormals, areaFaces, queue, newFaces):
    return calculateNewElementsOfQueue(queue, region.regionIndex, newFaces, region.proxyNormal, areaFaces, faceNormals)


"""
insert 'region' into 'regions'
"""
def InsertRegions(regions, insertRegion):
    regions[insertRegion.regionIndex] = insertRegion
    return regions


"""
calculate the area-weighted normal of all regions
"""
def GetProxy(proxys):
    for proxy in proxys.values():
        proxy.proxyNormal = GetProxyNormal(proxy.faceIndexes)
    return proxys


"""
Given a list of faces indices, calculate the area_weighted normal
"""
def GetProxyNormal(indexes):
    proxyNormal = np.array([0, 0, 0])
    for index in indexes:
        proxyNormal = np.add(proxyNormal, normalsGlobal[index])
    if proxyNormal[0] != 0 or proxyNormal[1] != 0 or proxyNormal[2] != 0:
        proxyNormal = proxyNormal / np.linalg.norm(proxyNormal)
    return proxyNormal


"""
calcualte L2-distance between all faces and all regions w.r.t normals -> proxys
"""
def GetProxySeed(proxys, faceNormals, areaFaces):
    regions = InitProxyList()
    for proxy in proxys.values():
        regionIndex = proxy.regionIndex
        faceIndexes = proxy.faceIndexes
        proxyNormal = proxy.proxyNormal

        tmp = MetricError(regionIndex,
                             faceIndexes,
                             faceNormals,
                             areaFaces,
                             proxyNormal)
        errors = []
        while not tmp.empty():
            errors.append(tmp.get())
        errors.sort(key=lambda x: -x.error)
        seedFaceIndex = errors.pop().index
        region = Proxy(proxy.regionIndex,
                       [seedFaceIndex],
                       proxyNormal=proxy.proxyNormal)
        regions = InsertRegions(regions,region)
    return regions

"""
Given a region, get face-this region proxys and put them into queue
"""
def UpdateQueue(region, faceNormals, areaFaces, queue, newFaces):
    regionIndex = region.regionIndex
    proxyNormal = region.proxyNormal
    calculateNewElementsOfQueue(queue, regionIndex, newFaces, proxyNormal, areaFaces, faceNormals)
    return queue

"""
Find  the region with biggest L2 distance
"""
def findWorst(queue):
    index = 0
    maxE = -np.inf
    for i in range(len(queue)):
        if queue[i].error > maxE:
            maxE = queue[i].error
            index = i
    return queue.pop(index)

"""
Distribue les faces aux différentes régions en fonction de leur proximité avec celles ci
"""
def AssignToRegion(faceNormals, areaFaces, adjacentFaces, regions, queue, assignedIndexes):
    globalQueue = []
    assignedIndexes = set(assignedIndexes)
    while not queue.empty():
        mostPriority = queue.get()
        faceIndex = mostPriority.index
        if faceIndex not in assignedIndexes:
            globalQueue.append(mostPriority)
            region = FindRegion(regions, mostPriority.regionIndex)
            region.faceIndexes.append(faceIndex)
            assignedIndexes.add(faceIndex)
            newAdjacentFaces = set(adjacentFaces[faceIndex])
            newAdjacentFaces -= assignedIndexes
            UpdateQueueNew(region,
                           faceNormals,
                           areaFaces,
                           queue,
                           newAdjacentFaces)

    try:
        worst = findWorst(globalQueue)
    except IndexError:
        randomReg = random.randrange(0,len(regions)-1)
        randomRegFace = random.randrange(0,len(regions[randomReg])-1)
        worst = QueueElement(0.0, regions[randomReg].regionIndex, regions[randomReg].faceIndexes[randomRegFace])
    return regions, worst

"""
repeat poping the top element from the queue until all faces are assigned to a region
"""
def AssignToWorstRegion(faceNormals, areaFaces, adjacentFaces, regions, queue, assignedIndexes, oldRegionFaces):
    regionDomain = frozenset(oldRegionFaces)
    assignedIndexes = set(assignedIndexes)
    while not queue.empty():
        mostPriority = queue.get()
        if mostPriority.index not in regionDomain:
            continue
        faceIndex = mostPriority.index
        if faceIndex not in assignedIndexes:
            regionIndex = mostPriority.regionIndex
            for region in regions.values():
                if regionIndex == region.regionIndex:
                    region.faceIndexes.append(faceIndex)
                    assignedIndexes.add(faceIndex)
                    s = set(adjacentFaces[faceIndex])
                    s &= regionDomain
                    s -= assignedIndexes
                    if s:
                        UpdateQueue(region,
                                    faceNormals,
                                    areaFaces,
                                    queue,
                                    s)


    return regions


def BuildQueue(regions, faceNormals, areaFaces, adjacentToFaces):
    assignedIndexes = []
    queue = PriorityQueue()
    for region in regions.values():
        seedIndex = region.faceIndexes[0]
        assignedIndexes.append(seedIndex)
        UpdateQueue(region,
                    faceNormals,
                    areaFaces,
                    queue,
                    adjacentToFaces[seedIndex])
    return queue, assignedIndexes


def SplitRegion(faceNormals, areaFaces, adjacentFaces, regions, worst):
    worstRegion = FindRegion(regions, worst.regionIndex)
    splitRegion_A = generateId()
    splitRegion_B = generateId()

    oldRegionFaces = worstRegion.faceIndexes
    seedIndex_A = oldRegionFaces[0]
    seedIndex_B = worst.index
    splitRegions = GetProxy(InsertRegions(InsertRegions(InitProxyList(), Proxy(splitRegion_A, [seedIndex_A])), Proxy(splitRegion_B, [seedIndex_B])))

    queue, assignedIndexes = BuildQueue(splitRegions,
                                        faceNormals,
                                        areaFaces,
                                        adjacentFaces)

    splitRegions = AssignToWorstRegion(faceNormals,
                                       areaFaces,
                                       adjacentFaces,
                                       splitRegions,
                                       queue,
                                       assignedIndexes,
                                       oldRegionFaces)
    return InsertRegions(InsertRegions(RemoveRegion(regions, worstRegion.regionIndex), splitRegions[splitRegion_A]), splitRegions[splitRegion_B])


def FindAdjacentRegions(faceEdges, regions):
    adjacentRegions = []
    regionsEdges = []
    for region in regions.values():
        regionIndex = region.regionIndex
        regionEdges = []
        for i in region.faceIndexes:
            regionEdges.extend(faceEdges[i])
        regionsEdges.append([regionIndex, set(regionEdges)])
    for region_A, region_B in itertools.combinations(regionsEdges, 2):
        if region_A[1].intersection(region_B[1]):
            adjacentRegions.append([region_A[0], region_B[0]])

    return adjacentRegions


def FindRegionsToCombine(regions, adjacentRegions, faceNormals, areaFaces):
    params = {
        "maxError": -np.inf,
        "regionsToCombine": None,
        "mergedRegion": None,
        "i": None
    }
    regionsToDelete = adjacentRegions[0]
    for i, adjacent in enumerate(adjacentRegions):
        region_A = FindRegion(regions, adjacent[0])
        region_B = FindRegion(regions, adjacent[1])
        mergedRegion = GetProxy(InsertRegions(InitProxyList(),Proxy(generateId(), region_A.faceIndexes + region_B.faceIndexes)))[str(ID_MESH_LAST)]
        proxyNormal = mergedRegion.proxyNormal
        params = {
            "maxError": params["maxError"],
            "regionsToCombine": params["regionsToCombine"],
            "mergedRegion": mergedRegion,
            "i": i
        }
        params = calculateNewElementsOfQueue([], 0, mergedRegion.faceIndexes, proxyNormal, areaFaces, faceNormals, True,
                                             params)
        if params["regionsToCombine"] == mergedRegion:
            regionsToDelete = adjacent
    return InsertRegions(RemoveRegion(RemoveRegion(regions, regionsToDelete[0]), regionsToDelete[1]),
                         params["regionsToCombine"])


def GrowSeeds(subFaceIndexes, faceVertexIndexes, vertices):
    verticesOfRegionByFace = [faceVertexIndexes[i] for i in subFaceIndexes]
    verticesOfRegion = set([i for sublist in verticesOfRegionByFace for i in sublist])
    mapa = dict(list(zip(verticesOfRegion, list(range(len(verticesOfRegion))))))
    subVertices = list(mapa.keys())
    newFaceIndexes = []
    for item in verticesOfRegionByFace:
        newFaceIndexes.append([mapa[i] for i in item])
    newVertices = {}
    for k, v in mapa.items():
        newVertices[v] = vertices[k]
    newVertices = list(newVertices.values())
    return np.array(newVertices), newFaceIndexes


"""
return mesh ID
"""
ID_MESH_LAST = 0

def generateId():
    global ID_MESH_LAST
    ID_MESH_LAST += 1
    return str(ID_MESH_LAST)


"""
return random RGB color
"""


def Randomcolor():
    return random.randint(0, 255) / 255, random.randint(0, 255) / 255, random.randint(0, 255) / 255


"""
return random nb regions
"""
def generateNRegions(mesh, nb, adjacency):
    listFaces = mesh.faces[:]
    regions = InitProxyList()
    faceDrawn = []
    for i in range(nb):
        face = random.randrange(len(listFaces))
        while face in faceDrawn:
            face = random.randrange(len(listFaces))
        regions = InsertRegions(regions,Proxy(generateId(), [face]))
        faceDrawn.append(face)
    return regions

def ndArrayToArray(ndA):
    retArr = []
    for v in ndA:
        retArr.append(list(v))
    return retArr

def arrayToNdArray(a):
    return np.array(a)

def proxyToJson(p):
    return {
        "regionIndex": p.regionIndex,
        "faceIndexes": p.faceIndexes,
        "proxyNormal": list(p.proxyNormal) if p.proxyNormal is not None else p.proxyNormal
    }

def jsonToProxy(j):
    return Proxy(j["regionIndex"], j["faceIndexes"], np.array(j["proxyNormal"]) if j["proxyNormal"] else j["proxyNormal"])

def meshToJson(m):
    return {
        "vertices": ndArrayToArray(m.vertices),
        "faces": m.faces,
        "edges": m.edges
    }

def jsonToMesh(j):
    mRet = Mesh(arrayToNdArray(j["vertices"]), j["faces"])
    mRet.edges = j["edges"]
    return mRet

def populateProxys(j):
    proxys = InitProxyList()
    for p in j:
        InsertRegions(proxys, jsonToProxy(p))
    return proxys

def saveState(stateName = None):
    global fileName
    jsonToSave = json.dumps({
        "nbExec": nbExec, # OK
        "vertsGlobal":ndArrayToArray(vertsGlobal),
        "facesGlobal":facesGlobal,
        "proxysGlobal":[proxyToJson(p) for p in proxysGlobal.values()],
        "normalsGlobal":normalsGlobal, # OK
        "meshGlobal":meshToJson(meshGlobal),
        "areasGlobal":areasGlobal,
        "edgesGlobal":edgesGlobal,
        "adjacencyGlobal":[list(s) for s in adjacencyGlobal],
        "nbProxys":nbProxys, # OK
        "ID_MESH_LAST": ID_MESH_LAST # OK
    })
    while stateName.endswith(".json"):
        stateName = stateName[:-5]
    if stateName is None or stateName == "name":
        tab = ["0","1","2","3","4","5","6","7","8","9","a","b","c","d","e","f"]
        stateName = ""
        for i in range(20):
            stateName += tab[random.randrange(0,15)]
        fileName = stateName
    with open(stateName + '.json', 'w') as outfile:
        json.dump(jsonToSave, outfile)

def loadState(stateName):
    global nbExec, vertsGlobal, facesGlobal, proxysGlobal, normalsGlobal, meshGlobal, areasGlobal, edgesGlobal, adjacencyGlobal, nbProxys, ID_MESH_LAST
    with open(stateName + '.json') as json_file:
        data = json.loads(json.load(json_file))
        nbExec = data["nbExec"]
        vertsGlobal = arrayToNdArray(data["vertsGlobal"])
        facesGlobal = data["facesGlobal"]
        proxysGlobal = populateProxys(data["proxysGlobal"])
        normalsGlobal = data["normalsGlobal"]
        meshGlobal = jsonToMesh(data["meshGlobal"])
        areasGlobal = data["areasGlobal"]
        edgesGlobal = data["edgesGlobal"]
        adjacencyGlobal = [set(s) for s in data["adjacencyGlobal"]]
        nbProxys = data["nbProxys"]
        ID_MESH_LAST = data["ID_MESH_LAST"]

def corpse():
    global nbExec, vertsGlobal, facesGlobal, proxysGlobal, normalsGlobal, meshGlobal, areasGlobal, edgesGlobal, adjacencyGlobal, nbProxys, fileName
    psim.PushItemWidth(150)
    psim.TextUnformatted("Execute VSA-algorithm")
    psim.Separator()

    changed, nbExec = psim.InputInt("Iteration times", nbExec, step=1, step_fast=3)
    psim.SameLine()
    if psim.Button("Execute"):
        newProxys = KMeans(
            nbExec,
            proxysGlobal,
            normalsGlobal,
            meshGlobal.vertices,
            meshGlobal.faces,
            areasGlobal,
            edgesGlobal,
            adjacencyGlobal
        )
        RefreshAllProxys(proxysGlobal, newProxys, meshGlobal)
        proxysGlobal = newProxys

    if psim.Button("Add regions"):
        # calcualte regions' area_weighted normals
        proxys = GetProxy(proxysGlobal)
        # calcualte L2-distance between all faces and all regions w.r.t normals -> proxys
        regions = GetProxySeed(proxys, normalsGlobal, areasGlobal)
        # put udpated proxys into priority queue
        queue, assignedIndexes = BuildQueue(regions, normalsGlobal, areasGlobal, adjacencyGlobal)
        regions, worst = AssignToRegion(normalsGlobal, areasGlobal, adjacencyGlobal, regions, queue, assignedIndexes)
        regions = SplitRegion(normalsGlobal, areasGlobal, adjacencyGlobal, regions, worst)
        RefreshAllProxys(proxysGlobal, regions, meshGlobal)
        proxysGlobal = regions
        nbProxys += 1

    psim.SameLine()
    if psim.Button("Delete region"):
        proxys = GetProxy(proxysGlobal)
        regions = GetProxySeed(proxys, normalsGlobal, areasGlobal)
        queue, assignedIndexes = BuildQueue(regions, normalsGlobal, areasGlobal, adjacencyGlobal)
        regions, worst = AssignToRegion(normalsGlobal, areasGlobal, adjacencyGlobal, regions, queue, assignedIndexes)
        adjacentRegions = FindAdjacentRegions(edgesGlobal, regions)
        regions = FindRegionsToCombine(regions, adjacentRegions, normalsGlobal, areasGlobal)
        RefreshAllProxys(proxysGlobal, regions, meshGlobal)
        proxysGlobal = regions
        nbProxys -= 1

    psim.SameLine()
    if psim.Button("Restart"):
        tmp = generateNRegions(meshGlobal, nbProxys, adjacencyGlobal)
        RefreshAllProxys(proxysGlobal, tmp, meshGlobal)
        proxysGlobal = tmp

    psim.Separator()
    psim.TextUnformatted("Save / load state")
    changed, fileName = psim.InputText("Filename", fileName)
    if psim.Button("Save state"):
        saveState(fileName)
        print("Saved !")
    psim.SameLine()
    if psim.Button("Load state"):
        RefreshAllProxys(proxysGlobal, InitProxyList(), meshGlobal)
        loadState(fileName)
        ps.register_surface_mesh("MAIN", vertsGlobal, facesGlobal, color=(0., 1., 0.), edge_color=(0., 0., 0.),
                                 edge_width=3)
        RefreshAllProxys(InitProxyList(), proxysGlobal, meshGlobal)
        ps.reset_camera_to_home_view()


nbExec = 1
vertsGlobal = None
facesGlobal = None
proxysGlobal = None
normalsGlobal = None
meshGlobal = None
areasGlobal = None
edgesGlobal = None
adjacencyGlobal = None
nbProxys = None
fileName = "name"

def InitProxyList():
    return {}

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='input mesh')
    parser.add_argument('--nbProxys', type=int, default=100, help='number of regions')
    args = parser.parse_args()
    global vertsGlobal, facesGlobal, proxysGlobal, normalsGlobal, meshGlobal, areasGlobal, edgesGlobal, adjacencyGlobal, nbProxys
    
    print(args)

    obj = load_obj(args.path, triangulate=True)
    vertsGlobal = obj.only_coordinates()
    facesGlobal = obj.only_faces()

    meshGlobal = Mesh(vertsGlobal, facesGlobal)
    st = time.time()
    normalsGlobal = meshGlobal.getAllFacesNormals()
    print("normals : ", time.time() - st)
    st = time.time()
    areasGlobal = meshGlobal.getAllFacesArea()
    print("areas : ", time.time() - st)
    st = time.time()
    edgesGlobal = meshGlobal.getAllFaceEdges()
    print("edges : ", time.time() - st)
    st = time.time()
    adjacencyGlobal = meshGlobal.getAllAdjacentFaces()
    print("adjacency : ", time.time() - st)
    print("Number of faces : ", len(areasGlobal))
    nbProxys = args.nbProxys
    proxysGlobal = generateNRegions(meshGlobal, nbProxys, adjacencyGlobal)
    RefreshAllProxys(InitProxyList(), proxysGlobal, meshGlobal)
    ps.init()
    ps.set_user_callback(corpse)
    ps.register_surface_mesh("MAIN", vertsGlobal, facesGlobal, color=(0., 1., 0.), edge_color=(0., 0., 0.),
                             edge_width=3)
    ps.show()


if __name__ == '__main__':
    main()