import glob
import os
import shutil
import stat
import numpy
import math
import platform

def make_project_directory():
    # Empty project directory
    home = os.path.expanduser("~")
    dirname = os.path.join(home, 'openWEC')
    calcname = os.path.join(dirname, 'Calculation')
    meshname = os.path.join(calcname, 'mesh')
    resname = os.path.join(calcname, 'results')
    nemohname = os.path.join(dirname, 'Nemoh')
    mooringname = os.path.join(dirname, 'Mooring')
    outputname = os.path.join(dirname, 'Output')
    othername = os.path.join(dirname, 'Other')
    if os.path.isdir(dirname):
        if os.path.isdir(calcname):
            if not os.path.isdir(meshname):
                os.mkdir(meshname)
            if not os.path.isdir(resname):
                os.mkdir(resname)
        else:
            os.mkdir(calcname)
        if not os.path.isdir(nemohname):
            os.mkdir(nemohname)
        if not os.path.isdir(mooringname):
            os.mkdir(mooringname)
        if not os.path.isdir(outputname):
            os.mkdir(outputname)
        if not os.path.isdir(othername):
            os.mkdir(othername)

    else:
        os.mkdir(dirname)
        os.mkdir(calcname)
        os.mkdir(nemohname)
        os.mkdir(meshname)
        os.mkdir(resname)
        os.mkdir(mooringname)
        os.mkdir(outputname)
        os.mkdir(othername)

    # Copy necessary files and stuff
    shutil.copy('./blankProject/Calculation/Solver.exe', calcname)
    shutil.copy('./blankProject/Calculation/input.txt', calcname)
    shutil.copy('./blankProject/Calculation/preProcessor.exe', calcname)
    shutil.copy('./blankProject/Calculation/Mesh.cal', calcname)
    shutil.copy('./blankProject/Calculation/Nemoh.cal', calcname)
    shutil.copy('./blankProject/Calculation/postProc', calcname)
    shutil.copy('./blankProject/Calculation/postProcO', calcname)
    shutil.copy('./blankProject/Calculation/meshL', calcname)
    shutil.copy('./blankProject/Calculation/meshO', calcname)
    shutil.copy('./blankProject/Calculation/solver', calcname)
    shutil.copy('./blankProject/Calculation/solverO', calcname)
    shutil.copy('./blankProject/Calculation/postProcessor.exe', calcname)
    shutil.copy('./blankProject/Calculation/ID.dat', calcname)
    shutil.copy('./blankProject/Calculation/Mesh.exe', calcname)
    shutil.copy('./blankProject/Calculation/preProc', calcname)
    shutil.copy('./blankProject/Calculation/preProcO', calcname)
    shutil.copy('./blankProject/Other/spec_test.dat', othername)
    shutil.copy('./blankProject/Mooring/lines_template.txt', mooringname)
    shutil.copy('./blankProject/Mooring/lines.txt', mooringname)

    # Give permissions to executables
    os.chmod(os.path.join(calcname, 'meshL'), stat.S_IRWXU)
    os.chmod(os.path.join(calcname, 'preProc'), stat.S_IRWXU)
    os.chmod(os.path.join(calcname, 'solver'), stat.S_IRWXU)
    os.chmod(os.path.join(calcname, 'postProc'), stat.S_IRWXU)
    os.chmod(os.path.join(calcname, 'meshO'), stat.S_IRWXU)
    os.chmod(os.path.join(calcname, 'preProcO'), stat.S_IRWXU)
    os.chmod(os.path.join(calcname, 'solverO'), stat.S_IRWXU)
    os.chmod(os.path.join(calcname, 'postProcO'), stat.S_IRWXU)


def clean_directory():
    current_directory = os.getcwd()
    os.chdir(os.path.join(os.path.join(os.path.expanduser('~'), 'openWEC'), 'Calculation'))
    file_list = glob.glob('./axisym*.dat')
    for fil in file_list:
        os.remove(fil)
    os.chdir('./mesh')
    file_list = os.listdir('./')
    for fil in file_list:
        os.remove(fil)
    os.chdir('..')
    os.chdir('./results')
    file_list = os.listdir('./')
    for fil in file_list:
        os.remove(fil)
    os.chdir('..')
    os.chdir('..')
    os.chdir('Nemoh')
    file_list = os.listdir('./')
    for fil in file_list:
        os.remove(fil)
    os.chdir(current_directory)


class Mesh:
    def __init__(self):
        # Define blank values
        self.np = 0
        self.nf = 0
        self.X = []
        self.Y = []
        self.Z = []
        self.P = []

    def combine_meshes(self, ob1, ob2):
        # Check for largest mesh
        if ob1.nf < ob2.nf:
            coin_test = ob1.make_coin()
            coin_target = ob2.make_coin()
        else:
            coin_test = ob2.make_coin()
            coin_target = ob1.make_coin()
        # Check for duplicate panels
        deletion_list = []
        for iF in range(numpy.size(coin_test[1, 1, :])):
            panel_test = coin_test[:, :, iF]
            for iFF in range(numpy.size(coin_target[1, 1, :])):
                panel_target = coin_target[:, :, iFF]
                if numpy.sum(panel_test == panel_target) == 12:
                    coin_target = numpy.delete(coin_target, iFF, 2)
                    deletion_list.append(iF)
        coin_test = numpy.delete(coin_test, deletion_list, 2)

        # Concatenate unique meshes
        coin = numpy.concatenate((coin_test, coin_target), axis=2)
        self.np = numpy.size(coin[1, 1, :]) * 4
        self.nf = numpy.size(coin[1, 1, :])
        self.X = numpy.zeros(numpy.size(coin[1, 1, :]) * 4)
        self.Y = numpy.zeros(numpy.size(coin[1, 1, :]) * 4)
        self.Z = numpy.zeros(numpy.size(coin[1, 1, :]) * 4)
        self.P = numpy.zeros((numpy.size(coin[1, 1, :]), 4), dtype=int)

        iP = 0
        for iF in range(numpy.size(coin[1, 1, :])):
            for iC in range(4):
                self.X[iP] = coin[0, iC, iF]
                self.Y[iP] = coin[1, iC, iF]
                self.Z[iP] = coin[2, iC, iF]
                iP += 1
            self.P[iF, 0] = 1 + iF * 4
            self.P[iF, 1] = 2 + iF * 4
            self.P[iF, 2] = 3 + iF * 4
            self.P[iF, 3] = 4 + iF * 4

    def make_coin(self):
        coin = numpy.zeros((3, 4, self.nf))
        for iF in range(self.nf):
            for iC in range(4):
                coin[0, iC, iF] = self.X[self.P[iF, iC] - 1]
                coin[1, iC, iF] = self.Y[self.P[iF, iC] - 1]
                coin[2, iC, iF] = self.Z[self.P[iF, iC] - 1]
        return coin

    def delete_horizontal_panels(self):
        coin = self.make_coin()
        apex = numpy.min(self.Z)
        zLoc = numpy.zeros(4)
        deletion_list = []

        # Check every panel for horizontality and higher position than lowest point
        for iP in range(self.nf):
            for iC in range(4):
                zLoc[iC] = coin[2, iC, iP]
            if numpy.abs(numpy.mean(zLoc) - zLoc[0]) < 0.001 and numpy.mean(zLoc) > apex:
                deletion_list.append(iP)

        # Delete selected panels
        coin = numpy.delete(coin, deletion_list, 2)

        # Remake mesh
        self.np = numpy.size(coin[1, 1, :]) * 4
        self.nf = numpy.size(coin[1, 1, :])
        self.X = numpy.zeros(numpy.size(coin[1, 1, :]) * 4)
        self.Y = numpy.zeros(numpy.size(coin[1, 1, :]) * 4)
        self.Z = numpy.zeros(numpy.size(coin[1, 1, :]) * 4)
        self.P = numpy.zeros((numpy.size(coin[1, 1, :]), 4), dtype=int)

        iP = 0
        for iF in range(numpy.size(coin[1, 1, :])):
            for iC in range(4):
                self.X[iP] = coin[0, iC, iF]
                self.Y[iP] = coin[1, iC, iF]
                self.Z[iP] = coin[2, iC, iF]
                iP += 1
            self.P[iF, 0] = 1 + (iF) * 4
            self.P[iF, 1] = 2 + (iF) * 4
            self.P[iF, 2] = 3 + (iF) * 4
            self.P[iF, 3] = 4 + (iF) * 4


def writeMesh(msh, filename):
    with open(filename, 'w') as f:
        f.write('{:d}\n'.format(msh.np))
        f.write('{:d}\n'.format(msh.nf))
        for iP in range(msh.np):
            f.write('  {:.7f}  {:.7f}  {:.7f}\n'.format(msh.X[iP], msh.Y[iP], msh.Z[iP]))
        for iF in range(msh.nf):
            f.write('  {:d}  {:d}  {:d}  {:d}\n'.format(msh.P[iF, 0], msh.P[iF, 1], msh.P[iF, 2], msh.P[iF, 3]))
        return None


class box:
    def __init__(self, length, width, height, cCor):
        self.length = length
        self.width = width
        self.height = height
        self.xC = cCor[0]
        self.yC = cCor[1]
        self.zC = cCor[2]
        self.name = 'box'
        self.panelize()
        self.translate(self.xC, self.yC, self.zC)

    def panelize(self):
        self.nf = 6
        self.np = 8
        self.X = numpy.array(
            [-self.length / 2.0, self.length / 2.0, -self.length / 2.0, self.length / 2.0, -self.length / 2.0,
             self.length / 2.0, -self.length / 2.0, self.length / 2.0])
        self.Y = numpy.array([self.width / 2.0, self.width / 2.0, self.width / 2.0, self.width / 2.0, -self.width / 2.0,
                              -self.width / 2.0, -self.width / 2.0, -self.width / 2.0])
        self.Z = numpy.array(
            [-self.height / 2.0, -self.height / 2.0, self.height / 2.0, self.height / 2.0, -self.height / 2.0,
             -self.height / 2.0, self.height / 2.0, self.height / 2.0])
        self.P = numpy.zeros([6, 4], dtype=int)
        self.P[0, :] = numpy.array([3, 4, 2, 1])
        self.P[1, :] = numpy.array([4, 8, 6, 2])
        self.P[2, :] = numpy.array([8, 7, 5, 6])
        self.P[3, :] = numpy.array([7, 3, 1, 5])
        self.P[4, :] = numpy.array([2, 6, 5, 1])
        self.P[5, :] = numpy.array([8, 4, 3, 7])
        # Define triangles for plotting
        self.trii = numpy.zeros([2 * self.nf, 3], dtype=int)
        iT = 0
        for iTr in range(self.nf):
            self.trii[iT, :] = [self.P[iTr, 0] - 1, self.P[iTr, 1] - 1, self.P[iTr, 2] - 1]
            self.trii[iT + 1, :] = [self.P[iTr, 0] - 1, self.P[iTr, 2] - 1, self.P[iTr, 3] - 1]
            iT += 2

    def translate(self, xT, yT, zT):
        self.X += xT
        self.Y += yT
        self.Z += zT

    def rotate(self, a1, a2, theta):
        R = numpy.zeros([3, 3])
        # Normal vector through origin
        u = a2[0] - a1[0]
        v = a2[1] - a1[1]
        w = a2[2] - a1[2]
        u = u / numpy.sqrt(u ** 2 + v ** 2 + w ** 2)
        v = v / numpy.sqrt(u ** 2 + v ** 2 + w ** 2)
        w = w / numpy.sqrt(u ** 2 + v ** 2 + w ** 2)
        # Translate mesh so that rotation axis starts from the origin
        self.X -= a1[0]
        self.Y -= a1[1]
        self.Z -= a1[2]

        # Rotation matrix
        R[0, 0] = u ** 2 + numpy.cos(theta) * (1 - u ** 2)
        R[0, 1] = u * v * (1 - numpy.cos(theta)) - w * numpy.sin(theta)
        R[0, 2] = u * w * (1 - numpy.cos(theta)) + v * numpy.sin(theta)
        R[1, 0] = u * v * (1 - numpy.cos(theta)) + w * numpy.sin(theta)
        R[1, 1] = v ** 2 + numpy.cos(theta) * (1 - v ** 2)
        R[1, 2] = v * w * (1 - numpy.cos(theta)) - u * numpy.sin(theta)
        R[2, 0] = w * u * (1 - numpy.cos(theta)) - v * numpy.sin(theta)
        R[2, 1] = w * v * (1 - numpy.cos(theta)) + u * numpy.sin(theta)
        R[2, 2] = w ** 2 + numpy.cos(theta) * (1 - w ** 2)

        for iP in range(self.np):
            p1 = numpy.array([self.X[iP], self.Y[iP], self.Z[iP]])
            p2 = numpy.dot(R, p1)
            self.X[iP] = p2[0]
            self.Y[iP] = p2[1]
            self.Z[iP] = p2[2]

        # Translate back to original position

        self.X += a1[0]
        self.Y += a1[1]
        self.Z += a1[2]

    def makeCoin(self):
        coin = numpy.zeros((3, 4, self.nf))
        for iF in range(self.nf):
            for iC in range(4):
                coin[0, iC, iF] = self.X[self.P[iF, iC] - 1]
                coin[1, iC, iF] = self.Y[self.P[iF, iC] - 1]
                coin[2, iC, iF] = self.Z[self.P[iF, iC] - 1]
        return coin


class cone:
    def __init__(self, diameter, height, cCor):
        self.diameter = diameter
        self.height = height
        self.xC = cCor[0]
        self.yC = cCor[1]
        self.zC = cCor[2]
        self.name = 'cone'
        self.panelize()
        self.translate(self.xC, self.yC, self.zC)

    def panelize(self):
        Ntheta = 18
        Nz = 3
        theta = [xx * 2 * numpy.pi / (Ntheta - 1) for xx in range(Ntheta)]
        self.nf = 0
        self.np = 0
        r = [0, self.diameter / 2.0, 0]
        z = [0, 0, -self.height]
        self.X = []
        self.Y = []
        self.Z = []
        self.P = numpy.zeros([(len(r) - 1) * (Ntheta - 1), 4], dtype=int)
        n = len(r)

        for iT in range(Ntheta):
            for iN in range(n):
                self.X.append(r[iN] * numpy.cos(theta[iT]))
                self.Y.append(r[iN] * numpy.sin(theta[iT]))
                self.Z.append(z[iN])
                self.np += 1

        iP = 0
        for iN in range(1, n):
            for iT in range(1, Ntheta):
                self.P[iP, 0] = iN + n * (iT - 1)
                self.P[iP, 1] = iN + 1 + n * (iT - 1)
                self.P[iP, 2] = iN + 1 + n * iT
                self.P[iP, 3] = iN + n * iT
                self.nf += 1
                iP += 1

        self.X = numpy.array(self.X)
        self.Y = numpy.array(self.Y)
        self.Z = numpy.array(self.Z)
        # Define triangles for plotting
        self.trii = numpy.zeros([2 * self.nf, 3], dtype=int)
        iT = 0
        for iTr in range(self.nf):
            self.trii[iT, :] = [self.P[iTr, 0] - 1, self.P[iTr, 1] - 1, self.P[iTr, 2] - 1]
            self.trii[iT + 1, :] = [self.P[iTr, 0] - 1, self.P[iTr, 2] - 1, self.P[iTr, 3] - 1]
            iT += 2

    def translate(self, xT, yT, zT):
        self.X += xT
        self.Y += yT
        self.Z += zT

    def rotate(self, a1, a2, theta):
        R = numpy.zeros([3, 3])
        # Normal vector through origin
        u = a2[0] - a1[0]
        v = a2[1] - a1[1]
        w = a2[2] - a1[2]
        u = u / numpy.sqrt(u ** 2 + v ** 2 + w ** 2)
        v = v / numpy.sqrt(u ** 2 + v ** 2 + w ** 2)
        w = w / numpy.sqrt(u ** 2 + v ** 2 + w ** 2)
        # Translate mesh so that rotation axis starts from the origin
        self.X -= a1[0]
        self.Y -= a1[1]
        self.Z -= a1[2]

        # Rotation matrix
        R[0, 0] = u ** 2 + numpy.cos(theta) * (1 - u ** 2)
        R[0, 1] = u * v * (1 - numpy.cos(theta)) - w * numpy.sin(theta)
        R[0, 2] = u * w * (1 - numpy.cos(theta)) + v * numpy.sin(theta)
        R[1, 0] = u * v * (1 - numpy.cos(theta)) + w * numpy.sin(theta)
        R[1, 1] = v ** 2 + numpy.cos(theta) * (1 - v ** 2)
        R[1, 2] = v * w * (1 - numpy.cos(theta)) - u * numpy.sin(theta)
        R[2, 0] = w * u * (1 - numpy.cos(theta)) - v * numpy.sin(theta)
        R[2, 1] = w * v * (1 - numpy.cos(theta)) + u * numpy.sin(theta)
        R[2, 2] = w ** 2 + numpy.cos(theta) * (1 - w ** 2)

        for iP in range(self.np):
            p1 = numpy.array([self.X[iP], self.Y[iP], self.Z[iP]])
            p2 = numpy.dot(R, p1)
            self.X[iP] = p2[0]
            self.Y[iP] = p2[1]
            self.Z[iP] = p2[2]

        # Translate back to original position

        self.X += a1[0]
        self.Y += a1[1]
        self.Z += a1[2]

    def makeCoin(self):
        coin = numpy.zeros((3, 4, self.nf))
        for iF in range(self.nf):
            for iC in range(4):
                coin[0, iC, iF] = self.X[self.P[iF, iC] - 1]
                coin[1, iC, iF] = self.Y[self.P[iF, iC] - 1]
                coin[2, iC, iF] = self.Z[self.P[iF, iC] - 1]
        return coin


class cylinder:
    def __init__(self, diameter, height, cCor):
        self.diameter = diameter
        self.height = height
        self.xC = cCor[0]
        self.yC = cCor[1]
        self.zC = cCor[2]
        self.name = 'cylinder'
        self.panelize()
        self.translate(self.xC, self.yC, self.zC)

    def panelize(self):
        Ntheta = 18
        Nz = 3
        theta = [xx * 2 * numpy.pi / (Ntheta - 1) for xx in range(Ntheta)]
        self.nf = 0
        self.np = 0
        r = [0, self.diameter / 2.0, self.diameter / 2.0, 0]
        z = [0, 0, -self.height, -self.height]
        self.X = []
        self.Y = []
        self.Z = []
        self.P = numpy.zeros([(len(r) - 1) * (Ntheta - 1), 4], dtype=int)
        n = len(r)

        for iT in range(Ntheta):
            for iN in range(n):
                self.X.append(r[iN] * numpy.cos(theta[iT]))
                self.Y.append(r[iN] * numpy.sin(theta[iT]))
                self.Z.append(z[iN])
                self.np += 1

        iP = 0
        for iN in range(1, n):
            for iT in range(1, Ntheta):
                self.P[iP, 0] = iN + n * (iT - 1)
                self.P[iP, 1] = iN + 1 + n * (iT - 1)
                self.P[iP, 2] = iN + 1 + n * iT
                self.P[iP, 3] = iN + n * iT
                self.nf += 1
                iP += 1

        self.X = numpy.array(self.X)
        self.Y = numpy.array(self.Y)
        self.Z = numpy.array(self.Z)
        # Define triangles for plotting
        self.trii = numpy.zeros([2 * self.nf, 3], dtype=int)
        iT = 0
        for iTr in range(self.nf):
            self.trii[iT, :] = [self.P[iTr, 0] - 1, self.P[iTr, 1] - 1, self.P[iTr, 2] - 1]
            self.trii[iT + 1, :] = [self.P[iTr, 0] - 1, self.P[iTr, 2] - 1, self.P[iTr, 3] - 1]
            iT += 2

    def translate(self, xT, yT, zT):
        self.X += xT
        self.Y += yT
        self.Z += zT

    def rotate(self, a1, a2, theta):
        R = numpy.zeros([3, 3])
        # Normal vector through origin
        u = a2[0] - a1[0]
        v = a2[1] - a1[1]
        w = a2[2] - a1[2]
        u = u / numpy.sqrt(u ** 2 + v ** 2 + w ** 2)
        v = v / numpy.sqrt(u ** 2 + v ** 2 + w ** 2)
        w = w / numpy.sqrt(u ** 2 + v ** 2 + w ** 2)
        # Translate mesh so that rotation axis starts from the origin
        self.X -= a1[0]
        self.Y -= a1[1]
        self.Z -= a1[2]

        # Rotation matrix
        R[0, 0] = u ** 2 + numpy.cos(theta) * (1 - u ** 2)
        R[0, 1] = u * v * (1 - numpy.cos(theta)) - w * numpy.sin(theta)
        R[0, 2] = u * w * (1 - numpy.cos(theta)) + v * numpy.sin(theta)
        R[1, 0] = u * v * (1 - numpy.cos(theta)) + w * numpy.sin(theta)
        R[1, 1] = v ** 2 + numpy.cos(theta) * (1 - v ** 2)
        R[1, 2] = v * w * (1 - numpy.cos(theta)) - u * numpy.sin(theta)
        R[2, 0] = w * u * (1 - numpy.cos(theta)) - v * numpy.sin(theta)
        R[2, 1] = w * v * (1 - numpy.cos(theta)) + u * numpy.sin(theta)
        R[2, 2] = w ** 2 + numpy.cos(theta) * (1 - w ** 2)

        for iP in range(self.np):
            p1 = numpy.array([self.X[iP], self.Y[iP], self.Z[iP]])
            p2 = numpy.dot(R, p1)
            self.X[iP] = p2[0]
            self.Y[iP] = p2[1]
            self.Z[iP] = p2[2]

        # Translate back to original position

        self.X += a1[0]
        self.Y += a1[1]
        self.Z += a1[2]

    def makeCoin(self):
        coin = numpy.zeros((3, 4, self.nf))
        for iF in range(self.nf):
            for iC in range(4):
                coin[0, iC, iF] = self.X[self.P[iF, iC] - 1]
                coin[1, iC, iF] = self.Y[self.P[iF, iC] - 1]
                coin[2, iC, iF] = self.Z[self.P[iF, iC] - 1]
        return coin


class hemicylinder:
    def __init__(self, diameter, height, cCor):
        self.diameter = diameter
        self.height = height
        self.xC = cCor[0]
        self.yC = cCor[1]
        self.zC = cCor[2]
        self.name = 'hemicylinder'
        self.panelize()
        self.translate(self.xC, self.yC, self.zC)

    def panelize(self):
        Ntheta = 18
        Nz = 3
        theta = [xx * numpy.pi / (Ntheta - 1) - numpy.pi / 2.0 for xx in range(Ntheta)]
        self.nf = 0
        self.np = 0
        r = [0, self.diameter / 2.0, self.diameter / 2.0, 0]
        z = [self.height / 2.0, self.height / 2.0, -self.height / 2.0, -self.height / 2.0]
        self.X = []
        self.Y = []
        self.Z = []
        self.P = numpy.zeros([(len(r) - 1) * (Ntheta - 1), 4], dtype=int)
        n = len(r)

        for iT in range(Ntheta):
            for iN in range(n):
                self.Z.append(-r[iN] * numpy.cos(theta[iT]))
                self.X.append(r[iN] * numpy.sin(theta[iT]))
                self.Y.append(z[iN])
                self.np += 1

        iP = 0
        for iN in range(1, n):
            for iT in range(1, Ntheta):
                self.P[iP, 3] = iN + n * (iT - 1)
                self.P[iP, 2] = iN + 1 + n * (iT - 1)
                self.P[iP, 1] = iN + 1 + n * iT
                self.P[iP, 0] = iN + n * iT
                self.nf += 1
                iP += 1

        self.X = numpy.array(self.X)
        self.Y = numpy.array(self.Y)
        self.Z = numpy.array(self.Z)
        # Define triangles for plotting
        self.trii = numpy.zeros([2 * self.nf, 3], dtype=int)
        iT = 0
        for iTr in range(self.nf):
            self.trii[iT, :] = [self.P[iTr, 0] - 1, self.P[iTr, 1] - 1, self.P[iTr, 2] - 1]
            self.trii[iT + 1, :] = [self.P[iTr, 0] - 1, self.P[iTr, 2] - 1, self.P[iTr, 3] - 1]
            iT += 2

    def translate(self, xT, yT, zT):
        self.X += xT
        self.Y += yT
        self.Z += zT

    def rotate(self, a1, a2, theta):
        R = numpy.zeros([3, 3])
        # Normal vector through origin
        u = a2[0] - a1[0]
        v = a2[1] - a1[1]
        w = a2[2] - a1[2]
        u = u / numpy.sqrt(u ** 2 + v ** 2 + w ** 2)
        v = v / numpy.sqrt(u ** 2 + v ** 2 + w ** 2)
        w = w / numpy.sqrt(u ** 2 + v ** 2 + w ** 2)
        # Translate mesh so that rotation axis starts from the origin
        self.X -= a1[0]
        self.Y -= a1[1]
        self.Z -= a1[2]

        # Rotation matrix
        R[0, 0] = u ** 2 + numpy.cos(theta) * (1 - u ** 2)
        R[0, 1] = u * v * (1 - numpy.cos(theta)) - w * numpy.sin(theta)
        R[0, 2] = u * w * (1 - numpy.cos(theta)) + v * numpy.sin(theta)
        R[1, 0] = u * v * (1 - numpy.cos(theta)) + w * numpy.sin(theta)
        R[1, 1] = v ** 2 + numpy.cos(theta) * (1 - v ** 2)
        R[1, 2] = v * w * (1 - numpy.cos(theta)) - u * numpy.sin(theta)
        R[2, 0] = w * u * (1 - numpy.cos(theta)) - v * numpy.sin(theta)
        R[2, 1] = w * v * (1 - numpy.cos(theta)) + u * numpy.sin(theta)
        R[2, 2] = w ** 2 + numpy.cos(theta) * (1 - w ** 2)

        for iP in range(self.np):
            p1 = numpy.array([self.X[iP], self.Y[iP], self.Z[iP]])
            p2 = numpy.dot(R, p1)
            self.X[iP] = p2[0]
            self.Y[iP] = p2[1]
            self.Z[iP] = p2[2]

        # Translate back to original position

        self.X += a1[0]
        self.Y += a1[1]
        self.Z += a1[2]

    def makeCoin(self):
        coin = numpy.zeros((3, 4, self.nf))
        for iF in range(self.nf):
            for iC in range(4):
                coin[0, iC, iF] = self.X[self.P[iF, iC] - 1]
                coin[1, iC, iF] = self.Y[self.P[iF, iC] - 1]
                coin[2, iC, iF] = self.Z[self.P[iF, iC] - 1]
        return coin


class sphere:
    def __init__(self, diameter, cCor):
        self.diameter = diameter
        self.xC = cCor[0]
        self.yC = cCor[1]
        self.zC = cCor[2]
        self.name = 'sphere'
        self.panelize()
        self.translate(self.xC, self.yC, self.zC)

    def panelize(self):
        Ntheta = 18
        Nthetad2 = int(Ntheta / 2)
        Nz = 3
        theta = [xx * 2 * numpy.pi / (Ntheta - 1) for xx in range(Ntheta)]
        phi = [xx * numpy.pi / (Ntheta / 2 - 1) for xx in range(Nthetad2)]
        self.nf = 0
        self.np = 0
        r = self.diameter / 2.0
        self.X = []
        self.Y = []
        self.Z = []
        self.P = numpy.zeros([(Ntheta - 1) * (Nthetad2 - 1), 4], dtype=int)

        for iT in range(Nthetad2):
            for iTT in range(Ntheta):
                self.X.append(r * numpy.cos(theta[iTT]) * numpy.sin(phi[iT]))
                self.Y.append(r * numpy.sin(theta[iTT]) * numpy.sin(phi[iT]))
                self.Z.append(r * numpy.cos(phi[iT]))
                self.np += 1

        iP = 0
        for iN in range(1, Ntheta):
            for iT in range(1, Nthetad2):
                self.P[iP, 3] = iN + Ntheta * (iT - 1)
                self.P[iP, 2] = iN + 1 + Ntheta * (iT - 1)
                self.P[iP, 1] = iN + 1 + Ntheta * iT
                self.P[iP, 0] = iN + Ntheta * iT
                self.nf += 1
                iP += 1
        self.X = numpy.array(self.X)
        self.Y = numpy.array(self.Y)
        self.Z = numpy.array(self.Z)
        # Define triangles for plotting
        self.trii = numpy.zeros([2 * self.nf, 3], dtype=int)
        iT = 0
        for iTr in range(self.nf):
            self.trii[iT, :] = [self.P[iTr, 0] - 1, self.P[iTr, 1] - 1, self.P[iTr, 2] - 1]
            self.trii[iT + 1, :] = [self.P[iTr, 0] - 1, self.P[iTr, 2] - 1, self.P[iTr, 3] - 1]
            iT += 2

    def translate(self, xT, yT, zT):
        self.X += xT
        self.Y += yT
        self.Z += zT

    def rotate(self, a1, a2, theta):
        R = numpy.zeros([3, 3])
        # Normal vector through origin
        u = a2[0] - a1[0]
        v = a2[1] - a1[1]
        w = a2[2] - a1[2]
        u = u / numpy.sqrt(u ** 2 + v ** 2 + w ** 2)
        v = v / numpy.sqrt(u ** 2 + v ** 2 + w ** 2)
        w = w / numpy.sqrt(u ** 2 + v ** 2 + w ** 2)
        # Translate mesh so that rotation axis starts from the origin
        self.X -= a1[0]
        self.Y -= a1[1]
        self.Z -= a1[2]

        # Rotation matrix
        R[0, 0] = u ** 2 + numpy.cos(theta) * (1 - u ** 2)
        R[0, 1] = u * v * (1 - numpy.cos(theta)) - w * numpy.sin(theta)
        R[0, 2] = u * w * (1 - numpy.cos(theta)) + v * numpy.sin(theta)
        R[1, 0] = u * v * (1 - numpy.cos(theta)) + w * numpy.sin(theta)
        R[1, 1] = v ** 2 + numpy.cos(theta) * (1 - v ** 2)
        R[1, 2] = v * w * (1 - numpy.cos(theta)) - u * numpy.sin(theta)
        R[2, 0] = w * u * (1 - numpy.cos(theta)) - v * numpy.sin(theta)
        R[2, 1] = w * v * (1 - numpy.cos(theta)) + u * numpy.sin(theta)
        R[2, 2] = w ** 2 + numpy.cos(theta) * (1 - w ** 2)

        for iP in range(self.np):
            p1 = numpy.array([self.X[iP], self.Y[iP], self.Z[iP]])
            p2 = numpy.dot(R, p1)
            self.X[iP] = p2[0]
            self.Y[iP] = p2[1]
            self.Z[iP] = p2[2]

        # Translate back to original position

        self.X += a1[0]
        self.Y += a1[1]
        self.Z += a1[2]

    def makeCoin(self):
        coin = numpy.zeros((3, 4, self.nf))
        for iF in range(self.nf):
            for iC in range(4):
                coin[0, iC, iF] = self.X[self.P[iF, iC] - 1]
                coin[1, iC, iF] = self.Y[self.P[iF, iC] - 1]
                coin[2, iC, iF] = self.Z[self.P[iF, iC] - 1]
        return coin


class hemisphere:
    def __init__(self, diameter, cCor):
        self.diameter = diameter
        self.xC = cCor[0]
        self.yC = cCor[1]
        self.zC = cCor[2]
        self.name = 'hemisphere'
        self.panelize()
        self.translate(self.xC, self.yC, self.zC)

    def panelize(self):
        Ntheta = 18
        theta = [xx * 2 * numpy.pi / (Ntheta - 1) for xx in range(Ntheta)]
        phi = [xx * numpy.pi / 2.0 / (Ntheta / 2 - 1) for xx in range(Ntheta / 2)]
        self.nf = 0
        self.np = 0
        r = self.diameter / 2.0
        self.X = []
        self.Y = []
        self.Z = []
        self.P = numpy.zeros([(Ntheta - 1) * (Ntheta / 2 - 1), 4], dtype=int)

        for iT in range(Ntheta / 2):
            for iTT in range(Ntheta):
                self.X.append(r * numpy.cos(theta[iTT]) * numpy.sin(phi[iT]))
                self.Y.append(r * numpy.sin(theta[iTT]) * numpy.sin(phi[iT]))
                self.Z.append(-r * numpy.cos(phi[iT]))
                self.np += 1

        iP = 0
        for iN in range(1, Ntheta):
            for iT in range(1, Ntheta / 2):
                self.P[iP, 0] = iN + Ntheta * (iT - 1)
                self.P[iP, 1] = iN + 1 + Ntheta * (iT - 1)
                self.P[iP, 2] = iN + 1 + Ntheta * iT
                self.P[iP, 3] = iN + Ntheta * iT
                self.nf += 1
                iP += 1

        self.X = numpy.array(self.X)
        self.Y = numpy.array(self.Y)
        self.Z = numpy.array(self.Z)
        # Define triangles for plotting
        self.trii = numpy.zeros([2 * self.nf, 3], dtype=int)
        iT = 0
        for iTr in range(self.nf):
            self.trii[iT, :] = [self.P[iTr, 0] - 1, self.P[iTr, 1] - 1, self.P[iTr, 2] - 1]
            self.trii[iT + 1, :] = [self.P[iTr, 0] - 1, self.P[iTr, 2] - 1, self.P[iTr, 3] - 1]
            iT += 2

    def translate(self, xT, yT, zT):
        self.X += xT
        self.Y += yT
        self.Z += zT

    def rotate(self, a1, a2, theta):
        R = numpy.zeros([3, 3])
        # Normal vector through origin
        u = a2[0] - a1[0]
        v = a2[1] - a1[1]
        w = a2[2] - a1[2]
        u = u / numpy.sqrt(u ** 2 + v ** 2 + w ** 2)
        v = v / numpy.sqrt(u ** 2 + v ** 2 + w ** 2)
        w = w / numpy.sqrt(u ** 2 + v ** 2 + w ** 2)
        # Translate mesh so that rotation axis starts from the origin
        self.X -= a1[0]
        self.Y -= a1[1]
        self.Z -= a1[2]

        # Rotation matrix
        R[0, 0] = u ** 2 + numpy.cos(theta) * (1 - u ** 2)
        R[0, 1] = u * v * (1 - numpy.cos(theta)) - w * numpy.sin(theta)
        R[0, 2] = u * w * (1 - numpy.cos(theta)) + v * numpy.sin(theta)
        R[1, 0] = u * v * (1 - numpy.cos(theta)) + w * numpy.sin(theta)
        R[1, 1] = v ** 2 + numpy.cos(theta) * (1 - v ** 2)
        R[1, 2] = v * w * (1 - numpy.cos(theta)) - u * numpy.sin(theta)
        R[2, 0] = w * u * (1 - numpy.cos(theta)) - v * numpy.sin(theta)
        R[2, 1] = w * v * (1 - numpy.cos(theta)) + u * numpy.sin(theta)
        R[2, 2] = w ** 2 + numpy.cos(theta) * (1 - w ** 2)

        for iP in range(self.np):
            p1 = numpy.array([self.X[iP], self.Y[iP], self.Z[iP]])
            p2 = numpy.dot(R, p1)
            self.X[iP] = p2[0]
            self.Y[iP] = p2[1]
            self.Z[iP] = p2[2]

        # Translate back to original position

        self.X += a1[0]
        self.Y += a1[1]
        self.Z += a1[2]

    def makeCoin(self):
        coin = numpy.zeros((3, 4, self.nf))
        for iF in range(self.nf):
            for iC in range(4):
                coin[0, iC, iF] = self.X[self.P[iF, iC] - 1]
                coin[1, iC, iF] = self.Y[self.P[iF, iC] - 1]
                coin[2, iC, iF] = self.Z[self.P[iF, iC] - 1]
        return coin


class wedge:
    def __init__(self, length, width, height, cCor):
        self.length = length
        self.width = width
        self.height = height
        self.xC = cCor[0]
        self.yC = cCor[1]
        self.zC = cCor[2]
        self.name = 'wedge'
        self.panelize()
        self.translate(self.xC, self.yC, self.zC)

    def panelize(self):
        self.nf = 6
        self.np = 8
        self.X = numpy.array(
            [0.0, 0.0, -self.length / 2.0, self.length / 2.0, 0.0, 0.0, -self.length / 2.0, self.length / 2.0])
        self.Y = numpy.array([self.width / 2.0, self.width / 2.0, self.width / 2.0, self.width / 2.0, -self.width / 2.0,
                              -self.width / 2.0, -self.width / 2.0, -self.width / 2.0])
        self.Z = numpy.array([-self.height, -self.height, 0.0, 0.0, -self.height, -self.height, 0.0, 0.0])
        self.P = numpy.zeros([6, 4], dtype=int)
        self.P[0, :] = numpy.array([3, 4, 2, 1])
        self.P[1, :] = numpy.array([4, 8, 6, 2])
        self.P[2, :] = numpy.array([8, 7, 5, 6])
        self.P[3, :] = numpy.array([7, 3, 1, 5])
        self.P[4, :] = numpy.array([2, 6, 5, 1])
        self.P[5, :] = numpy.array([8, 4, 3, 7])
        # Define triangles for plotting
        self.trii = numpy.zeros([2 * self.nf, 3], dtype=int)
        iT = 0
        for iTr in range(self.nf):
            self.trii[iT, :] = [self.P[iTr, 0] - 1, self.P[iTr, 1] - 1, self.P[iTr, 2] - 1]
            self.trii[iT + 1, :] = [self.P[iTr, 0] - 1, self.P[iTr, 2] - 1, self.P[iTr, 3] - 1]
            iT += 2

    def translate(self, xT, yT, zT):
        self.X += xT
        self.Y += yT
        self.Z += zT

    def rotate(self, a1, a2, theta):
        R = numpy.zeros([3, 3])
        # Normal vector through origin
        u = a2[0] - a1[0]
        v = a2[1] - a1[1]
        w = a2[2] - a1[2]
        u = u / numpy.sqrt(u ** 2 + v ** 2 + w ** 2)
        v = v / numpy.sqrt(u ** 2 + v ** 2 + w ** 2)
        w = w / numpy.sqrt(u ** 2 + v ** 2 + w ** 2)
        # Translate mesh so that rotation axis starts from the origin
        self.X -= a1[0]
        self.Y -= a1[1]
        self.Z -= a1[2]

        # Rotation matrix
        R[0, 0] = u ** 2 + numpy.cos(theta) * (1 - u ** 2)
        R[0, 1] = u * v * (1 - numpy.cos(theta)) - w * numpy.sin(theta)
        R[0, 2] = u * w * (1 - numpy.cos(theta)) + v * numpy.sin(theta)
        R[1, 0] = u * v * (1 - numpy.cos(theta)) + w * numpy.sin(theta)
        R[1, 1] = v ** 2 + numpy.cos(theta) * (1 - v ** 2)
        R[1, 2] = v * w * (1 - numpy.cos(theta)) - u * numpy.sin(theta)
        R[2, 0] = w * u * (1 - numpy.cos(theta)) - v * numpy.sin(theta)
        R[2, 1] = w * v * (1 - numpy.cos(theta)) + u * numpy.sin(theta)
        R[2, 2] = w ** 2 + numpy.cos(theta) * (1 - w ** 2)

        for iP in range(self.np):
            p1 = numpy.array([self.X[iP], self.Y[iP], self.Z[iP]])
            p2 = numpy.dot(R, p1)
            self.X[iP] = p2[0]
            self.Y[iP] = p2[1]
            self.Z[iP] = p2[2]

        # Translate back to original position

        self.X += a1[0]
        self.Y += a1[1]
        self.Z += a1[2]

    def makeCoin(self):
        coin = numpy.zeros((3, 4, self.nf))
        for iF in range(self.nf):
            for iC in range(4):
                coin[0, iC, iF] = self.X[self.P[iF, iC] - 1]
                coin[1, iC, iF] = self.Y[self.P[iF, iC] - 1]
                coin[2, iC, iF] = self.Z[self.P[iF, iC] - 1]
        return coin


class pyramid:
    def __init__(self, length, width, height, cCor):
        self.length = length
        self.width = width
        self.height = height
        self.xC = cCor[0]
        self.yC = cCor[1]
        self.zC = cCor[2]
        self.name = 'pyramid'
        self.panelize()
        self.translate(self.xC, self.yC, self.zC)

    def panelize(self):
        self.nf = 6
        self.np = 8
        self.X = numpy.array(
            [0.0, 0.0, -self.length / 2.0, self.length / 2.0, 0.0, 0.0, -self.length / 2.0, self.length / 2.0])
        self.Y = numpy.array(
            [0.0, 0.0, self.width / 2.0, self.width / 2.0, 0.0, 0.0, -self.width / 2.0, -self.width / 2.0])
        self.Z = numpy.array([-self.height, -self.height, 0.0, 0.0, -self.height, -self.height, 0.0, 0.0])
        self.P = numpy.zeros([6, 4], dtype=int)
        self.P[0, :] = numpy.array([3, 4, 2, 1])
        self.P[1, :] = numpy.array([4, 8, 6, 2])
        self.P[2, :] = numpy.array([8, 7, 5, 6])
        self.P[3, :] = numpy.array([7, 3, 1, 5])
        self.P[4, :] = numpy.array([5, 6, 5, 1])
        self.P[5, :] = numpy.array([8, 4, 3, 7])
        # Define triangles for plotting
        self.trii = numpy.zeros([2 * self.nf, 3], dtype=int)
        iT = 0
        for iTr in range(self.nf):
            self.trii[iT, :] = [self.P[iTr, 0] - 1, self.P[iTr, 1] - 1, self.P[iTr, 2] - 1]
            self.trii[iT + 1, :] = [self.P[iTr, 0] - 1, self.P[iTr, 2] - 1, self.P[iTr, 3] - 1]
            iT += 2

    def translate(self, xT, yT, zT):
        self.X += xT
        self.Y += yT
        self.Z += zT

    def rotate(self, a1, a2, theta):
        R = numpy.zeros([3, 3])
        # Normal vector through origin
        u = a2[0] - a1[0]
        v = a2[1] - a1[1]
        w = a2[2] - a1[2]
        u = u / numpy.sqrt(u ** 2 + v ** 2 + w ** 2)
        v = v / numpy.sqrt(u ** 2 + v ** 2 + w ** 2)
        w = w / numpy.sqrt(u ** 2 + v ** 2 + w ** 2)
        # Translate mesh so that rotation axis starts from the origin
        self.X -= a1[0]
        self.Y -= a1[1]
        self.Z -= a1[2]

        # Rotation matrix
        R[0, 0] = u ** 2 + numpy.cos(theta) * (1 - u ** 2)
        R[0, 1] = u * v * (1 - numpy.cos(theta)) - w * numpy.sin(theta)
        R[0, 2] = u * w * (1 - numpy.cos(theta)) + v * numpy.sin(theta)
        R[1, 0] = u * v * (1 - numpy.cos(theta)) + w * numpy.sin(theta)
        R[1, 1] = v ** 2 + numpy.cos(theta) * (1 - v ** 2)
        R[1, 2] = v * w * (1 - numpy.cos(theta)) - u * numpy.sin(theta)
        R[2, 0] = w * u * (1 - numpy.cos(theta)) - v * numpy.sin(theta)
        R[2, 1] = w * v * (1 - numpy.cos(theta)) + u * numpy.sin(theta)
        R[2, 2] = w ** 2 + numpy.cos(theta) * (1 - w ** 2)

        for iP in range(self.np):
            p1 = numpy.array([self.X[iP], self.Y[iP], self.Z[iP]])
            p2 = numpy.dot(R, p1)
            self.X[iP] = p2[0]
            self.Y[iP] = p2[1]
            self.Z[iP] = p2[2]

        # Translate back to original position

        self.X += a1[0]
        self.Y += a1[1]
        self.Z += a1[2]

    def makeCoin(self):
        coin = numpy.zeros((3, 4, self.nf))
        for iF in range(self.nf):
            for iC in range(4):
                coin[0, iC, iF] = self.X[self.P[iF, iC] - 1]
                coin[1, iC, iF] = self.Y[self.P[iF, iC] - 1]
                coin[2, iC, iF] = self.Z[self.P[iF, iC] - 1]
        return coin


class torus:
    def __init__(self, diamOut, diamIn, cCor):
        self.diamOut = diamOut
        self.diamIn = diamIn
        self.xC = cCor[0]
        self.yC = cCor[1]
        self.zC = cCor[2]
        self.name = 'torus'
        self.panelize()
        self.translate(self.xC, self.yC, self.zC)

    def panelize(self):
        Ntheta = 18
        Nphi = 18
        theta = [xx * 2 * numpy.pi / (Ntheta - 1) for xx in range(Ntheta)]
        phi = [xx * 2 * numpy.pi / (Nphi - 1) for xx in range(Nphi)]
        self.nf = 0
        self.np = 0
        self.X = []
        self.Y = []
        self.Z = []
        R = self.diamOut / 2.0
        r = self.diamIn / 2.0

        for iT in range(Ntheta):
            for iP in range(Nphi):
                self.X.append((R + r * numpy.cos(theta[iT])) * numpy.cos(phi[iP]))
                self.Y.append((R + r * numpy.cos(theta[iT])) * numpy.sin(phi[iP]))
                self.Z.append(r * numpy.sin(theta[iT]))
                self.np += 1

        self.nf = (Ntheta - 1) * (Nphi - 1)
        self.P = numpy.zeros([self.nf, 4], dtype=int)
        iPan = 0
        for iT in range(Ntheta - 1):
            for iP in range(Nphi - 1):
                self.P[iPan, 0] = iP + iT * Nphi + 1
                self.P[iPan, 1] = iP + 1 + iT * Nphi + 1
                self.P[iPan, 2] = iP + 1 + Ntheta + iT * Nphi + 1
                self.P[iPan, 3] = iP + Ntheta + iT * Nphi + 1
                iPan += 1

        self.X = numpy.array(self.X)
        self.Y = numpy.array(self.Y)
        self.Z = numpy.array(self.Z)
        # Define triangles for plotting
        self.trii = numpy.zeros([2 * self.nf, 3], dtype=int)
        iT = 0
        for iTr in range(self.nf):
            self.trii[iT, :] = [self.P[iTr, 0] - 1, self.P[iTr, 1] - 1, self.P[iTr, 2] - 1]
            self.trii[iT + 1, :] = [self.P[iTr, 0] - 1, self.P[iTr, 2] - 1, self.P[iTr, 3] - 1]
            iT += 2

    def translate(self, xT, yT, zT):
        self.X += xT
        self.Y += yT
        self.Z += zT

    def rotate(self, a1, a2, theta):
        R = numpy.zeros([3, 3])
        # Normal vector through origin
        u = a2[0] - a1[0]
        v = a2[1] - a1[1]
        w = a2[2] - a1[2]
        u = u / numpy.sqrt(u ** 2 + v ** 2 + w ** 2)
        v = v / numpy.sqrt(u ** 2 + v ** 2 + w ** 2)
        w = w / numpy.sqrt(u ** 2 + v ** 2 + w ** 2)
        # Translate mesh so that rotation axis starts from the origin
        self.X -= a1[0]
        self.Y -= a1[1]
        self.Z -= a1[2]

        # Rotation matrix
        R[0, 0] = u ** 2 + numpy.cos(theta) * (1 - u ** 2)
        R[0, 1] = u * v * (1 - numpy.cos(theta)) - w * numpy.sin(theta)
        R[0, 2] = u * w * (1 - numpy.cos(theta)) + v * numpy.sin(theta)
        R[1, 0] = u * v * (1 - numpy.cos(theta)) + w * numpy.sin(theta)
        R[1, 1] = v ** 2 + numpy.cos(theta) * (1 - v ** 2)
        R[1, 2] = v * w * (1 - numpy.cos(theta)) - u * numpy.sin(theta)
        R[2, 0] = w * u * (1 - numpy.cos(theta)) - v * numpy.sin(theta)
        R[2, 1] = w * v * (1 - numpy.cos(theta)) + u * numpy.sin(theta)
        R[2, 2] = w ** 2 + numpy.cos(theta) * (1 - w ** 2)

        for iP in range(self.np):
            p1 = numpy.array([self.X[iP], self.Y[iP], self.Z[iP]])
            p2 = numpy.dot(R, p1)
            self.X[iP] = p2[0]
            self.Y[iP] = p2[1]
            self.Z[iP] = p2[2]

        # Translate back to original position

        self.X += a1[0]
        self.Y += a1[1]
        self.Z += a1[2]

    def makeCoin(self):
        coin = numpy.zeros((3, 4, self.nf))
        for iF in range(self.nf):
            for iC in range(4):
                coin[0, iC, iF] = self.X[self.P[iF, iC] - 1]
                coin[1, iC, iF] = self.Y[self.P[iF, iC] - 1]
                coin[2, iC, iF] = self.Z[self.P[iF, iC] - 1]
        return coin


wdir = os.path.join(os.path.expanduser("~"), 'openWEC')


# Used Functions
def createMeshAxi(r, z, n, dtheta):
    print("1")
    nx = 0
    thetaR = range(0, dtheta)
    theta = [xx * math.pi / (dtheta - 1) for xx in thetaR]
    # write mesh file
    wpath = os.path.join(wdir, 'Calculation', 'mesh')
    if not os.path.exists(wpath):
        os.makedirs(wpath)
    fid = open(os.path.join(wpath, 'axisym'), 'w')
    values = str(n * dtheta) + '\n'
    fid.write(values)
    values = str((n - 1) * (dtheta - 1)) + '\n'
    fid.write(values)
    # calculate coordinates of mesh nodes
    for iT in range(0, dtheta):
        for iN in range(0, n):
            x = r[iN] * math.cos(theta[iT])
            y = r[iN] * math.sin(theta[iT])
            zz = z[iN]
            values = str(x) + '\t' + str(y) + '\t' + str(zz) + '\n'
            fid.write(values)
            nx = nx + 1
    # calculate connections
    nf = 0
    for iN in range(1, n):
        for iT in range(1, dtheta):
            NN1 = iN + n * (iT - 1)
            NN2 = iN + 1 + n * (iT - 1)
            NN3 = iN + 1 + n * iT
            NN4 = iN + n * iT
            values = str(NN1) + '\t' + str(NN2) + '\t' + str(NN3) + '\t' + str(NN4) + '\n'
            fid.write(values)
            nf = nf + 1

            # close mesh file
    fid.close()


def createMeshFull(n, X):
    print("2")
    nx = 0
    # write mesh file
    wpath = os.path.join(wdir, 'Calculation', 'mesh')
    if not os.path.exists(wpath):
        os.makedirs(wpath)
    fid = open(os.path.join(wpath, 'axisym'), 'w')
    values = str(n * 4) + '\n'
    fid.write(values)
    values = str(n) + '\n'
    fid.write(values)
    # calculate coordinates of mesh nodes and connections
    for pan in X:
        for point in pan:
            x = point[0]
            y = point[1]
            z = point[2]
            values = '{0:E} {1:E} {2:E} \n'.format(x, y, z)
            fid.write(values)
            nx += 1
    for iC in range(n):
        values = '{0:g} {1:g} {2:g} {3:g} \n'.format(4 * iC + 1, 4 * iC + 2, 4 * iC + 3, 4 * iC + 4)
        fid.write(values)
    # close mesh file
    fid.close()


def createMeshOpt(cG, nPanels, nsym, rho=1025.0, g=9.81, nbody=1, xG=0.0):
    print("3")
    curPath = os.getcwd()
    calPath = os.path.join(wdir, 'Calculation', 'Mesh.cal')
    if nbody == 1:
        fid = open(calPath, 'w')
        fid.write('axisym\n')
        fid.write('{:d}\n'.format(nsym))
        fid.write('0. 0.\n')
        value = '{0:f} {1:f} {2:f} \n'.format(cG[0], cG[1], cG[2])
        fid.write(value)
        fid.write(str(nPanels) + '\n')
        fid.write('2\n0.\n1.\n')
        fid.write('{0:f}\n'.format(rho))
        fid.write('{0:f}\n'.format(9.81))
        fid.close()
        os.chdir(os.path.join(wdir, 'Calculation'))
        if platform.system() == 'Windows':
            os.system('Mesh.exe')
        elif platform.system() == 'Darwin':
            os.system('./meshO')
        else:
            os.system('./meshL')
        os.chdir(curPath)
    else:
        for iB in range(nbody):
            fid = open(calPath, 'w')
            fid.write('axisym{:d}\n'.format(iB + 1))
            fid.write('{:d}\n'.format(nsym))
            fid.write('0. 0.\n')
            value = '{0:f} {1:f} {2:f} \n'.format(cG[0], cG[1], cG[2])
            fid.write(value)
            fid.write(str(nPanels) + '\n')
            fid.write('2\n0.\n1.\n')
            fid.write('{0:f}\n'.format(rho))
            fid.write('{0:f}\n'.format(9.81))
            fid.close()
            os.chdir(os.path.join(wdir, 'Calculation'))
            if platform.system() == 'Windows':
                os.system('Mesh.exe')
            elif platform.system() == 'Darwin':
                os.system('./meshO')
            else:
                os.system('./meshL')
            os.chdir(curPath)


def openParkFile(fname):
    print("4")
    parkPath = os.path.join(wdir, 'Other', 'parkconfig.dat')
    if os.path.isfile(parkPath):
        with open(fname) as f:
            allData = f.readlines()
        nrCoord = int(allData[0])
        coordList = []
        for iC in range(nrCoord):
            xLoc = float(allData[iC + 1].split()[0])
            yLoc = float(allData[iC + 1].split()[1])
            coordList.append(numpy.array([xLoc, yLoc]))
    return coordList


def makeArray(coordList):
    print("5")
    meshFile = os.path.join(wdir, 'Calculation', 'mesh', 'axisym.dat')
    baseMesh = numpy.loadtxt(meshFile, skiprows=1)
    nPoint = int(numpy.max(baseMesh[:, 0]))
    nPanel = len(baseMesh[nPoint + 1::, 0]) - 1
    panels = baseMesh[nPoint + 1:-1, :].astype(int)
    xBase = baseMesh[0:nPoint, 1]
    yBase = baseMesh[0:nPoint, 2]
    for iB in range(len(coordList)):
        xTr = coordList[iB][0]
        yTr = coordList[iB][1]
        meshFile = os.path.join(wdir, 'Calculation', 'mesh', 'axisym{:d}.dat'.format(iB + 1))
        with open(meshFile, 'w') as f:
            f.write('                    2          0\n')
            for iP in range(nPoint):
                f.write('             {0:d}             {1:f}             {2:f}             {3:f}\n'.format(iP + 1,
                                                                                                            xBase[
                                                                                                                iP] + xTr,
                                                                                                            yBase[
                                                                                                                iP] + yTr,
                                                                                                            baseMesh[
                                                                                                                iP, 3]))
            f.write('             0          0.00          0.00          0.00\n')
            for iP in range(nPanel):
                f.write('               {0:d}               {1:d}               {2:d}               {3:d}\n'.format(
                    panels[iP, 0], panels[iP, 1], panels[iP, 2], panels[iP, 3]))
            f.write(
                '               {0:d}               {1:d}               {2:d}               {3:d}\n'.format(0, 0, 0, 0))
        infoFile = os.path.join(wdir, 'Calculation', 'mesh', 'axisym{:d}_info.dat'.format(iB + 1))
        with open(infoFile, 'w') as f:
            f.write('    {0:d}     {1:d} Number of points and number of panels'.format(nPoint, nPanel))


def writeCalFile(rhoW, depW, omega, zG, dof, aO={}, nbody=1, xG=[0.0]):
    # In case of array simulation, do stuff
    if aO['parkCheck']:
        fname = os.path.join(wdir, 'Other', 'parkconfig.dat')
        coordList = openParkFile(fname)
        nbody = len(coordList)
        makeArray(coordList)
    # Read info on the mesh
    nrNode = [0] * nbody
    nrPanel = [0] * nbody
    for iB in range(nbody):
        if nbody == 1:
            infoName = os.path.join(wdir, 'Calculation', 'mesh', 'axisym_info.dat')
            f1 = open(infoName, 'r')
        else:
            infoName = os.path.join(wdir, 'Calculation', 'mesh', 'axisym{:d}_info.dat'.format(iB + 1))
            f1 = open(infoName, 'r')
        lineInfo = f1.readline()
        lineInfo = lineInfo.split()
        nrNode[iB] = int(lineInfo[0])
        nrPanel[iB] = int(lineInfo[1])
        f1.close()
    # Read advanced options if there are any
    dirCheck = aO['dirCheck']
    irfCheck = aO['irfCheck']
    kochCheck = aO['kochCheck']
    fsCheck = aO['fsCheck']

    # Create the Nemoh calibration file
    calFile = os.path.join(wdir, 'Calculation', 'Nemoh.cal')
    fid = open(calFile, 'w')
    fid.write('--- Environment ---\n')
    fid.write(str(rhoW) + '				! RHO 			! KG/M**3 	! Fluid specific volume\n')
    fid.write('9.81				! G			! M/S**2	! Gravity\n')
    fid.write(str(depW) + '				! DEPTH			! M		! Water depth\n')
    fid.write('0.	0.			! XEFF YEFF		! M		! Wave measurement point\n')
    fid.write('--- Description of floating bodies ---\n')
    fid.write('{:d}				! Number of bodies\n'.format(nbody))
    for iB in range(nbody):
        fid.write('--- Body {:d} ---\n'.format(iB + 1))
        if nbody == 1:
            fid.write('axisym.dat			! Name of mesh file\n')
        else:
            fid.write('axisym{:d}.dat			! Name of mesh file\n'.format(iB + 1))
        fid.write(str(nrNode[iB]) + '\t' + str(nrPanel[iB]) + '			! Number of points and number of panels\n')
        fid.write('{:d}				! Number of degrees of freedom\n'.format(sum(dof)))
        for iDof in range(len(dof)):
            if (iDof == 0 and dof[iDof] == 1):
                fid.write('1 1. 0.	0. 0. 0. 0.		! Surge\n')
            elif (iDof == 1 and dof[iDof] == 1):
                fid.write('1 0. 1.	0. 0. 0. 0.		! Sway\n')
            elif (iDof == 2 and dof[iDof] == 1):
                fid.write('1 0. 0. 1. 0. 0. 0.		! Heave\n')
            elif (iDof == 3 and dof[iDof] == 1):
                fid.write('2 1. 0. 0. {0:f} 0. {1:f}		! Roll about CdG\n'.format(xG[iB], zG))
            elif (iDof == 4 and dof[iDof] == 1):
                fid.write('2 0. 1. 0. {0:f} 0. {1:f}		! Pitch about CdG\n'.format(xG[iB], zG))
            elif (iDof == 5 and dof[iDof] == 1):
                fid.write('2 0. 0. 1. {0:f} 0. {1:f}		! Yaw about CdG\n'.format(xG[iB], zG))
        fid.write('{:d}				! Number of resulting generalised forces\n'.format(sum(dof)))
        for iDof in range(len(dof)):
            if (iDof == 0 and dof[iDof] == 1):
                fid.write('1 1. 0.	0. 0. 0. 0.		! Force in X direction\n')
            elif (iDof == 1 and dof[iDof] == 1):
                fid.write('1 0. 1.	0. 0. 0. 0.		! Force in Y direction\n')
            elif (iDof == 2 and dof[iDof] == 1):
                fid.write('1 0. 0. 1. 0. 0. 0.		! Force in Z direction\n')
            elif (iDof == 3 and dof[iDof] == 1):
                fid.write('2 1. 0. 0. {0:f} 0. {1:f}		! Roll Moment about CdG\n'.format(xG[iB], zG))
            elif (iDof == 4 and dof[iDof] == 1):
                fid.write('2 0. 1. 0. {0:f} 0. {1:f}		! Pitch Moment about CdG\n'.format(xG[iB], zG))
            elif (iDof == 5 and dof[iDof] == 1):
                fid.write('2 0. 0. 1. {0:f} 0. {1:f}		! Yaw Moment about CdG\n'.format(xG[iB], zG))
        fid.write('0				! Number of lines of additional information\n')

    fid.write('--- Load cases to be solved ---\n')
    fid.write(str(omega[0]) + '\t' + str(omega[1]) + '\t' + str(
        omega[2]) + '		! Number of wave frequencies, Min, and Max (rad/s)\n')
    if dirCheck:
        fid.write(str(aO['dirStep']) + '\t' + str(aO['dirStart']) + '\t' + str(
            aO['dirStop']) + '		! Number of wave directions, Min and Max (degrees)\n')
    else:
        fid.write('1	0.	0.		! Number of wave directions, Min and Max (degrees)\n')
    fid.write('--- Post processing ---\n')
    if irfCheck:
        fid.write('1' + '\t' + str(aO['irfStep']) + '\t' + str(aO[
                                                                   'irfDur']) + '\t\t! IRF 				! IRF calculation (0 for no calculation), time step and duration\n')
    else:
        fid.write(
            '0' + '\t0.01\t20.\t\t! IRF 				! IRF calculation (0 for no calculation), time step and duration\n')
    fid.write('0				! Show pressure\n')
    if kochCheck:
        fid.write(str(aO['kochStep']) + '\t' + str(aO['kochStart']) + '\t' + str(aO[
                                                                                     'kochStop']) + '		! Kochin function 		! Number of directions of calculation (0 for no calculations), Min and Max (degrees)\n')
    else:
        fid.write(
            '0	0.	180.		! Kochin function 		! Number of directions of calculation (0 for no calculations), Min and Max (degrees)\n')
    if fsCheck:
        fid.write(str(aO['fsDeltaX']) + '\t' + str(aO['fsDeltaY']) + '\t' + str(aO['fsLengthX']) + '\t' + str(aO[
                                                                                                                  'fsLengthY']) + '	! Free surface elevation 	! Number of points in x direction (0 for no calcutions) and y direction and dimensions of domain in x and y direction	\n')
    else:
        fid.write(
            '0	2	1000.	2.	! Free surface elevation 	! Number of points in x direction (0 for no calcutions) and y direction and dimensions of domain in x and y direction	\n')

    fid.close()
    return nbody


def runNemoh(nbody=1):
    print("modified")
    runDir = os.path.join(wdir, 'Calculation')
    curDir = os.getcwd()
    os.chdir(runDir)
    if nbody == 1:
        shutil.copyfile('./mesh/axisym.dat', 'axisym.dat')
    else:
        for iB in range(nbody):
            shutil.copyfile('./mesh/axisym{:d}.dat'.format(iB + 1), 'axisym{:d}.dat'.format(iB + 1))
    if platform.system() == 'Windows':
        os.system('preProcessor.exe')
        os.system('Solver.exe')
        os.system('postProcessor.exe')
    elif platform.system() == 'Darwin':
        os.system('./preProcO')
        os.system('./solverO')
        os.system('./postProcO')
    else:
        os.system('./preProc')
        os.system('./solver')
        os.system('./postProc')
    os.chdir(curDir)


def postNemoh(dof):
    print("8")
    # Open IRF file
    irfFile = os.path.join(wdir, 'Calculation', 'results', 'IRF.tec')
    with open(irfFile, 'r') as f:
        irfRaw = f.readlines()
        if dof == 'heave':
            strPat = 'DoF    1'
            irfInd = 2
        else:
            strPat = 'DoF    3'
            irfInd = 6
        for iL in range(0, len(irfRaw)):
            irfInfo = irfRaw[iL]
            test = irfInfo.find(strPat)
            if test != -1:
                indStart = iL
                irfInfo = irfInfo.split()
                irfNrlong = irfInfo[3]
                indComma = irfNrlong.index(',')
                irfNr = int(irfNrlong[0:indComma])
    # Arrange data in IRF file
    timeIRF = [0] * irfNr
    dataIRF = [0] * irfNr
    for iL in range(indStart + 1, irfNr + indStart + 1):
        irfLine = irfRaw[iL].split()
        timeIRF[iL - indStart - 1] = float(irfLine[0])
        dataIRF[iL - indStart - 1] = float(irfLine[irfInd])
