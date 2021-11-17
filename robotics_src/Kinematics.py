"""
Kinematics Module - Contains code for:
- Forward Kinematics, from a set of DH parameters to a serial linkage arm with callable forward kinematics
- Inverse Kinematics
- Jacobian

John Morrell, August 24 2021
Tarnarmour@gmail.com
"""

import numpy as np
import sympy as sp
import mpmath as mp
from robotics_src import hw_transforms as tr


def dh2A(dh, joint_type="r", q=sp.Symbol("q"), convention='standard', radians=True):
    """
    A = dh2A(dh, joint_type="r", q=sp.Symbol("q"), convention='standard')
    Description:
    Accepts one link of dh parameters and returns a homogeneous transform representing the transform from link i to link i+1

    Parameters:
    dh - 1 x 4 list or iterable of floats or sympy symbols, dh parameter table for one transform from link i to link i+1,
    in the order [d theta a alpha]
    q - sympy symbol, sympy symbol representing actuator input
    convention - string, 'standard' for standard dh convention, 'modified' for modified dh convention, 
    !!! modified not yet implemented !!!
    radians - bool, if false will assume theta and alpha are in degrees

    Returns:
    A - 4x4 sympy matrix representing the transform from one link to the next
    """
    # Convert to radians if needed
    if not radians:
        dh = [dh[0], mp.radians(dh[1]), dh[2], mp.radians(dh[3])]

    a = dh[2]
    alpha = dh[3]
    # If the joint is revolute, the actuator will change theta, while if it is prismatic it will affect d
    if joint_type == 'p':
        d = dh[0] + q
        theta = dh[1]
    else:
        d = dh[0]
        theta = dh[1] + q
    
    # See eq. (2.52), pg. 64
    A = sp.Matrix([[sp.cos(theta), -sp.sin(theta)*sp.cos(alpha), sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
                [sp.sin(theta), sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
                [0, sp.sin(alpha), sp.cos(alpha), d],
                [0, 0, 0, 1]])
    
    return A

class SerialArm:
    """
    Serial Arm - A class designed to simulate a serial linkage robotic arm
    Attributes:
    n - int, the number of links in the arm
    joint_type - list, a list of strings ('r' or 'p') representing whether each joint is prismatic or revolute (e.g. joint_type = ['r', 'r', 'p'])
    base - Sympy Matrix, Transform from the base to the first joint frame, defaults to identity
    tip - Sympy Matrix 4x4, Transform from the last joint frame to the tool tip
    is_symbolic - bool, if True indicates the arm's dh parameters have at least some symbolic elements defining the arm

    Methods:
    fk - returns the transform from one link to another
    jacob - returns the Jacobian at a given point on the arm
    jacoba - returns analytic jacobian at a given point
    ik - returns the joint coordinates to move the end effector to a specific point
    """
    def __init__(self, dh, joint_type=None, base=sp.eye(4), tip=sp.eye(4), radians=True, joint_limits=None):
        """
        arm = SerialArm(dh, joint_type=None, base=sp.eye(4), tip=sp.eye(4))
        Description:
        Constructor

        Parameters:
        dh - list or iterable of length 4 lists or iterables, each element is a list holding dh parameters for one link of the robot arm in the order [d theta a alpha]
        joint_type - list of strings, each string is either 'r' for revolute joint or 'p' for prismatic joint, for each link
        base - 4x4 sympy matrix representing the transform from base to first joint
        tip - 4x4 sympy matrix representing the transfrom from last joint to tool tip
        radians - bool, if False will assume dh parameters are given in degrees when valid

        Returns:
        arm - Instance of the SerialArm class
        """
        self.n = len(dh)
        self.joint_type = joint_type
        self.is_symbolic = False
        self.dh = []
        # Check if any values in DH are symbolic, and remake as a nested list for consistency
        for i in range(self.n):
            self.dh.append([dh[i][0], dh[i][1], dh[i][2], dh[i][3]])
            for j in range(4):
                if isinstance(dh[i][j], sp.Symbol):
                    self.is_symbolic = True

        self.transforms = []
        self.base = base
        self.tip = tip
        # Check if any values in base or tip are symbolic
        for i in range(4):
            for j in range(4):
                if isinstance(self.base[i,j], sp.Symbol) or isinstance(self.base[i,j], sp.Symbol):
                    self.is_symbolic = True
        
        if joint_type is None:
            self.joint_type = ['r'] * self.n
        
        for i in range(self.n):
            symbolic_tf = dh2A(dh[i], self.joint_type[i])
            self.transforms.append(symbolic_tf)

        q_sym = sp.symbols('q0:'+str(self.n), real=True)

        if not self.is_symbolic:
            fk_sym = self.fk(q_sym, base=base, tip=tip)
            fk_func = sp.lambdify(q_sym, fk_sym, modules='numpy')
            self.fk_fast = fk_func

            jacob_sym = self.jacob(q_sym)
            self.jacob_fast = sp.lambdify(q_sym, jacob_sym, modules='numpy')

        self.joint_limits = []
        if joint_limits is None:
            for i in range(self.n):
                if self.joint_type[i] == 'r':
                    self.joint_limits.append((-np.pi, np.pi))
                else:
                    self.joint_limits.append((-np.inf, np.inf))
        elif len(joint_limits) == 1:
            for i in range(self.n):
                self.joint_limits.append(joint_limits[0])
        else:
            for i in range(self.n):
                self.joint_limits.append(joint_limits[i])

    def clamp_q(self, q):
        q_out = sp.zeros(len(q), 1)
        for i in range(len(q)):
            if q[i] > self.joint_limits[i][1]:
                q_out[i, 0] = self.joint_limits[i][1]
            elif q[i] < self.joint_limits[i][0]:
                q_out[i, 0] = self.joint_limits[i][0]
            else:
                q_out[i, 0] = q[i]

        return q_out

    def subs(self, symbols, values):
        """
        new_arm = arm.subs(symbols, values)
        Description:
        subs works like the sympy function subs, returning a new SerialArm object with any sympy symbols in 'symbols' 
        replaced with their corresponding 'values'

        Parameters:
        symbols: list or iterable of sympy symbol variables to be replaced
        values: list or iterable of sympy symbols or floats that corresponds to symbols

        Returns:
        new_arm - SerialArm object identical to arm but with all symbolic values replaced.
        """

        subslist = []
        for i in range(len(symbols)):
            subslist.append([symbols[i], values[i]])

        dh = self.dh
        for i in range(self.n):
            for j in range(4):
                if not isinstance(dh[i][j], (float, int)):
                    dh[i][j] = dh[i][j].subs(subslist)
        
        base = self.base.subs(subslist)
        tip = self.tip.subs(subslist)

        return SerialArm(dh, self.joint_type, base, tip)
        
    def fk(self, q, index=-1, base=True, tip=True):
        """
        A = arm.fk(q, index=-1, base=True, tip=True)
        Description: 
        Returns the transform from one link to another given a set of joint inputs q

        Parameters:
        q - list or iterables of sympy symbols or floats which represent the joint actuator inputs to the arm
        index - integer or list of two integers. If a list of two integers, the first integer represents the starting JOINT 
        (with 0 as the first joint and n as the last joint) and the second integer represents the ending FRAME
        If one integer is given only, then the integer represents the ending Frame and the FK is calculated as starting from 
        the first joint
        base - bool, if True then if index starts from 0 the base transform will also be included
        tool - bool, if true and if the index ends at the nth frame then the tool transform will be included
        """

        if isinstance(index, (list, tuple)):
            start_link = index[0]
            end_link = index[1]
        else:
            start_link = 0
            end_link = index
        
        if end_link == -1:
            end_link = self.n
        elif end_link > self.n:
            print("WARNING: Ending index greater than number of joints")
            return None

        if start_link < 0:
            print("WARNING: Starting index less than zero")
            return None
        
        if base and start_link == 0:
            A = self.base
            start_link = 0
        else:
            A = sp.eye(4)
        
        # For each transform, get the transform by substituting q[i] into the transforms list, then post multiply
        for i in range(start_link, end_link):
            A = A * self.transforms[i].subs({"q":q[i]})
        
        if tip and end_link == self.n:
            A = A * self.tip
        
        return A.evalf()

    # def fk_fast(self, q):
    #     return q
    
    def jacob(self, q, index=-1, tip=True, base=True):
        """
        J = arm.jacob(q, index=-1)
        Description: 
        Returns the geometric jacobian of the arm in a given configuration

        Parameters:
        q - list of sympy symbols or floats, joint actuator inputs
        index - integer, which joint frame to give the jacobian at

        Returns:
        J - sympy matrix 6xN, jacobian of the robot arm
        """

        if index == -1:
            index = self.n
        elif index > self.n:
            print("WARNING: Index greater than number of joints")
            return None

        J = sp.zeros(6, index)
        Te = self.fk(q, index, base=base, tip=tip)
        Pe = Te[0:3, 3]
        for i in range(index):
            if self.joint_type[i] == 'r':
                T = self.fk(q, i, base=base, tip=tip)
                z_axis = T[0:3, 2]
                P = T[0:3, 3]
                J[0:3, i] = z_axis.cross(Pe - P)
                J[3:6, i] = z_axis
            else:
                T = self.fk(q, i, base=base, tip=tip)
                z_axis = T[0:3, 2]
                P = T[0:3, 3]
                J[0:3, i] = z_axis
                J[3:6, i] = sp.Matrix([[0.0], [0.0], [0.0]])
        return J

        # note: transforming and shifting jacobians should also be here as a seperate function
        # def jacob_shift(J, point, etc)
        # Maybe do once with a jacobian class, see if benefits outway complexity
        # Analytic jacobian

        # def IK(self, pose, method):

    def jacoba(self, q, index=-1, tip=True, base=True, symbolic=True, representation='rpy', h=1e-12):
        """
        J = arm.jacoba(q, index=-1, tip=True, base=True, symbolic=True, representation='rpy')
        Description:
        Returns the analytic jacobian of the point at index at the configuration q

        Parameters:
        q - list or iterable floats or sympy symbols, the joint configuration
        index - integer, which joint to return the jacobian for
        tip - bool, if true then if the index points to the end of the arm the jacobian for the tool tip will be returned
        base - bool, if true then the base transform will be used in the FK algorithm
        symbolic - bool, if true then sympy will attempt to symbolically differentiate the pose, then substitute q. This can be very computationally intensive
        representation - string, which representation of pose to use. 'rpy' for roll pitch yaw, 'quaternion' for quaternion, 'axis' for axis angle
        h - float, step size for non-symbolic finite difference derivative

        Returns:
        J - 6xN or 7xN sympy matrix, depending on which representation is used
        """
        if index == -1:
            index = self.n
        elif index > self.n:
            print("WARNING: Index greater than number of joints")
            return None

        if symbolic:
            symbol_string = "Q0"
            for i in range(1, self.n):
                symbol_string = symbol_string + " Q" + str(i)
            q_sym = sp.symbols(symbol_string)
            A = self.fk(q_sym, index=index, tip=tip, base=base)
            
            t = A[0:3, 3]
            R = tr.SO3(A[0:3, 0:3])

            if representation == 'rpy':
                w = R.rpy()
            elif representation == 'quaternion':
                w = R.quaternion()
            elif representation == 'axis':
                w = R.axis()
            elif representation == 'planar':
                t = A[0:2, 3]
                w = sp.Matrix(1, 1, [R.rpy()[0,0].evalf()])
            else:
                w = None
                print("Invalid value for representation in jacoba")
                return None

            pose = t.col_join(w)

            J = pose.jacobian(q_sym)
            subslist = []
            for i in range(self.n):
                subslist.append([q_sym[i], q[i]])
            J_eval = sp.N(J.subs(subslist), 4)
            return J_eval.as_mutable()

        else:
            n = self.n

            if representation == 'rpy':
                r = 6
            elif representation == 'quaternion':
                r = 7
            elif representation == 'axis':
                r = 7
            elif representation == 'planar':
                r = 3
            else:
                r = 6

            def get_pose(q):
                # A = self.fk(q, index=index, tip=tip, base=base).evalf()
                A = sp.Matrix(self.fk_fast(float(q[0]), float(q[1]), float(q[2]), float(q[3]), float(q[4])))
                t = A[0:3, 3]
                R = tr.SO3(A[0:3, 0:3])

                if representation == 'rpy':
                    w = R.rpy()
                elif representation == 'quaternion':
                    w = R.quaternion()
                elif representation == 'axis':
                    w = R.axis()
                elif representation == 'planar':
                    t = A[0:2, 3]
                    w = sp.Matrix(1, 1, [R.rpy()[0,0].evalf()])
                else:
                    w = None
                    print("Invalid value for representation in jacoba")
                    return None
                pose = t.col_join(w)
                return pose

            J = sp.zeros(r, n)

            for i in range(n):
                delta_q = sp.zeros(n, 1)
                delta_q[i] = 1e-10
                J[:, i] = (get_pose(q + delta_q) - get_pose(q - delta_q)) * 0.5 * 1e10

            return J



    def ik(self, T_target, q0=None, force_attempt=False, try_hard=False, method='pinv', tol=1e-3, k_damping=0.1, step_size=0.01, max_iter=100, representation='rpy', min_step=0.01):
        """
        qt, status, flag = arm.IK(A_target, q0=None, try_hard=False, tool=False, method='pinv'):
        Description:
        Searches for a set of joint angles that positions the end effector or tool at the target location

        Parameters: 
        A_target - 4x4 sympy matrix, the target location transform, expressed in the base frame of the robot arm
        q0 - n length iterable of floats, the initial configuration of the arm, defaults to q_i = 0
        force_attempt - bool, if true IK will try to force a solution even if initial checks consider the target to be out of the workspace
        try_hard - bool, if true IK will retry with multiple starting configurations until a successful solution is found
        tool - bool, if true the arm will try to position the tool instead of the end effector transform on the target
        method - string, which method to use. 'pinv' = psuedo-inverse jacobian, 'jt' = jacobian-transpose virtual wrench, 'min' = scipy minimization

        Returns:
        qt - length n list of floats or sympy symbols, the solution to the IK problem. If no solution is found qt will be the closet found point
        status - bool, true indicates successful IK solution, false indicates unsuccessful
        flag - string, indicates details about the termination of the IK algorithm, e.g. 'successful', 'above iteration limit', etc.
        """

        # Fill in q if none given, and convert to Matrix
        if not isinstance(q0, sp.Matrix):
            if q0 is None:
                q = sp.Matrix([0.0] * self.n)
            else:
                q = sp.Matrix(q0)
        else:
            q = q0

        # Check if the arm is symbolically defined; a symbolic arm cannot be solved for by numerical methods
        if self.is_symbolic:
            print("WARNING: Cannot evaluate IK for symbolic arm!")
            return q0, False, "Failed: Symbolic Arm"

        # Try basic check for if the target is in the workspace.
        # Maximum length of the arm is sum(sqrt(d_i^2 + a_i^2)), distance to target is norm(A_t)
        maximum_reach = 0
        for i in range(self.n):  # Add max length of each link
            maximum_reach = maximum_reach + sp.sqrt(self.dh[i][0]**2 + self.dh[i][2]**2)

        pt = T_target[0:3, 3]  # Find distance to target
        target_distance = sp.sqrt(pt[0]**2 + pt[1]**2 + pt[2]**2)

        if target_distance > maximum_reach and not force_attempt:
            print("WARNING: Target outside of reachable workspace!")
            return q, False, "Failed: Out of workspace"
        else:
            if target_distance > maximum_reach:
                print("Target out of workspace, but finding closest solution anyway")
            else:
                print("Target passes naive reach test, distance is {:.1} and max reach is {:.1}".format(float(target_distance), float(maximum_reach)))

        # Define convenient function for getting the chosen representation of pose from the rotation matrix
        def t2pose(T):
            R = tr.SO3(T[0:3, 0:3])
            p = T[0:3, 3]
            if representation == 'rpy':
                w = R.rpy()
            elif representation == 'quaternion':
                w = R.quaternion()
            elif representation == 'axis':
                w = R.axis()
            elif representation == 'planar':
                p = p[0:2,:]
                w = sp.Matrix(1, 1, [R.rpy()[0,0].evalf()])
            else:
                w = sp.Matrix([])
            return p.col_join(w)

        phit = t2pose(T_target)  # Calculate once to save time

        # Define function to get error in pose for each representation of pose
        def pose2error(qc):

            # Tc = self.fk(qc)
            Tc = sp.Matrix(self.fk_fast(float(qc[0]), float(qc[1]), float(qc[2]), float(qc[3]), float(qc[4])))
            phic = t2pose(Tc)

            e = sp.zeros(phit.shape[0], phit.shape[1])
            e[0:3,:] = phit[0:3,:] - phic[0:3,:]

            if representation == 'rpy':
                e[3:6,:] = phit[3:6,:] - phic[3:6,:]
            elif representation == 'quaternion':
                e[3:7,:] = phit[3:7,:] - phic[3:7,:]
            elif representation == 'axis':
                e[3:7,:] = phit[3:7,:] - phic[3:7,:]

            return e.evalf()

        # For convenience define a norm function to compare error
        def norm(x):
            s = 0
            if isinstance(x, (float, int, sp.Symbol)):
                return x
            for i in range(len(x)):
                s += x[i]**2
            return sp.N(sp.sqrt(s))  # Evaluate to stop sympy from blowing up symbolic terms

        flag = "Success, with {} iterations, residual error norm is {:.3}"
        status = True
        ################ Pseudo Inverse Method ##################
        if method == 'pinv':
            # Parameters:
            k = k_damping
            step = step_size
            iter = max_iter

            e = pose2error(q)
            count = 0

            while norm(e) > tol and count < iter:

                # Get analytic jacobian
                J = self.jacoba(q, representation=representation, symbolic=False)
                # Jtest = self.jacoba(q, representation=representation)

                r = J.shape[0]
                n = J.shape[1]

                # At this point we can determine what kind of solution we are looking for. Let n be the number of joints
                # and r is the dimensionality of the task space. Then J is an (r x n) matrix. There are 3 cases to
                # consider:
                if r == n:
                    # 1) r = n
                    # Interpretation: J is a square matrix, our task space dimensionality is the same as our joint space
                    # dimensionality

                    if J.rank() == r:
                        # 1.1) rank(J) = r
                        # Interpretation: J is full rank and invertible, and we can solve very simply using the inverse
                        # of the Jacobian.

                        dq = J.inv() @ e
                        dq = dq / dq.norm() * e.norm() * step
                        qnew = q + dq
                        print("Case 1.1\n")

                    else:
                        # 1.2) rank(J) < r
                        # Interpretation: J is not full rank, not invertible, but is the right shape. We can damp the
                        # Jacobian to ensure it is invertible in the neighbborhood of singularities. This will make the
                        # solution a bit slower and will not lead to solutions if target is outside of the range of the
                        # Jacobian.

                        J_damped = J.T @ (J @ J.T + sp.eye(n) * k_damping**2).inv()
                        dq = J_damped @ e
                        dq = dq / dq.norm() * e.norm() * step
                        qnew = q + dq
                        print("Case 1.2\n")

                if r > n:
                    # 2) r > n
                    # Interpretation: J is a 'tall' matrix, with less joints than task space dimensionality. The range
                    # of the Jacobian does not cover the whole task space, most likely the target cannot actually be
                    # reached by the arm. Instead of an exact solution, we seek a Least Squares Solution of the form
                    # qdot = (J'J)^-1 J' e (left psuedo-inverse).

                    if J.rank() == n:
                        # 2.1) rank(J) = n
                        # Interpretation: J is not invertible but it is full column rank, so we can use the solution
                        # described above

                        Jdag = (J.T @ J).inv() @ J.T
                        dq = Jdag @ e
                        dq = dq / dq.norm() * e.norm() * step
                        qnew = q + dq
                        print("Case 2.1\n")

                    else:
                        # 2.2) rank(J) < n
                        # Interpretation: Not only does R(J) not cover the task space, but J is not full column rank
                        # and (J'J) will not be invertible. To solve this we use the damped solution.

                        Jdag = (J.T @ J + sp.eye(n) * k_damping**2).inv() @ J.T
                        dq = Jdag @ e
                        dq = dq / dq.norm() * e.norm() * step
                        qnew = q + dq
                        print("Case 2.2\n")

                if r < n:
                    # 3) r < n
                    # Interpretation: J is a 'wide' matrix, joint space higher dimensionality than the task space. The
                    # robot arm is redundant, and it is possible to generate internal movements in the arm without
                    # moving the end effector. The IK problem is under-constrained, and we are free to specify
                    # additional goals to modifiy the solution.

                    if J.rank() == r:
                        # 3.1) rank(J) = r
                        # Interpretation: J has full row rank, which means we can use the right pseudo-inverse for the
                        # solution.

                        W = sp.eye(n)
                        Winv = W

                        Jdag = J.T @ (J @ J.T).inv()
                        dq = Jdag @ e
                        dq = dq / dq.norm() * e.norm() * step
                        qnew = q + dq
                        print("Case 3.1\n")

                    else:
                        # 3.2) rank(J) < r
                        # Interpretation: J is not full row rank, therefore we need to damp the answer to make J J'
                        # invertible
                        Jdag = J.T @ (J @ J.T + sp.eye(r) * k_damping**2)
                        dq = Jdag @ e
                        dq = dq / dq.norm() * e.norm() * step
                        qnew = q + dq
                        print("Case 3.2\n")

                # If we're stepping in the direction of the Jacobian, we have to move towards the
                # target. If the distance INCREASES, then we must have overshot. Reduce the step
                # size and try again. If step is good then reset step size.
                if norm(pose2error(qnew)) + tol / 2 > norm(e) and False:
                    step = step * 0.5
                    if step < min_step:
                        flag = 'Failure to converge, terminating because minimum step size reached after {} iterations, residual error norm is {:.3}'
                        status = False
                        break
                else:
                    q = qnew
                    step = step_size

                    for i in range(self.n):  # wrap q angles to be within [0 2*pi]
                        if q[i] > 2 * sp.pi:
                            d = q[i] / (2 * sp.pi)
                            q[i] = q[i] - 2 * sp.pi * sp.Integer(d)
                        if q[i] < 0:
                            d = q[i] / (2 * sp.pi)
                            q[i] = q[i] - 2 * sp.pi * (sp.Integer(d) - 1)

                    q = sp.N(q).as_mutable()
                    e = pose2error(q)

                count += 1
                if count == iter:
                    flag = 'Failure, did not converge within {} iterations, residual error norm is {:.3}'
                    status = False


            qf = list(q)

            flag = flag.format(count, norm(e))

            return qf,  status, flag
        
        if method == 'jt':
            # Jacobian transpose - virtual wrench method
            qf = q0
            status = True
            flag = 'Success'
            return qf,  status, flag 

        if method == 'min':
            # minimization method
            qf = q0
            status = True
            flag = 'Success'
            return qf,  status, flag 

    def ik2(self, target, q0=None, method='jt', force=False, tol=1e-6, K=None, kd=0.0001, dt=1, max_iter=100):
        """
        (qf, ef, iter) = arm.ik2(target, q0=None, method='jt', force=False, tol=1e-6, K=None)
        Description:
            Returns a solution to the inverse kinematics problem for a 6 dof arm, finding
            joint angles corresponding to the position (x y z coords) of target

        Args:
            target: 4x4 sympy matrix or length 3 iterable. If a matrix, target is the
            SE3 transform to the target location. If an iterable, contains only the
            [x y z] coordinates of the target location

            q0: length 6 iterable of initial joint coordinates, defaults to q=0 (which is
            often a singularity - other starting positions are recommended)

            method: String describing which IK algorithm to use. Options include:
                - 'pinv': damped pseudo-inverse solution, qdot = J_dag * e * dt, where
                J_dag = J.T * (J * J.T + K)^-1
                - 'jt': jacobian transpose method, qdot = J.T * K * e

            force: Boolean, if True will attempt to solve even if a naive reach check
            determines the target to be outside the reach of the arm

            tol: float, tolerance in the norm of the error in pose

            K: 3x3 sympy matrix or float. If K is a float, it is converted to K = eye(3)*K
            internally. If pinv is the method, K is the damping matrix used for the pseudo-inverse.
            If jt is the method, K is the positive definite gain matrix.

            max_iter: maximum attempts before giving up.

        Returns:
            qf: 6x1 sympy matrix of final joint values. If IK fails to converge the last set
            of joint angles is still returned

            ef: 3x1 sympy vector of the final error

            count: int, number of iterations

            flag: bool, true indicates successful IK solution and false unsuccessful
        """
        # Fill in q if none given, and convert to Matrix
        if not isinstance(q0, sp.Matrix):
            if q0 is None:
                q = sp.Matrix([0.0] * self.n)
            else:
                q = sp.Matrix(q0)
        else:
            q = q0

        # Check if the arm is symbolically defined; a symbolic arm cannot be solved for by numerical methods
        if self.is_symbolic:
            print("WARNING: Cannot evaluate IK for symbolic arm!")
            return q0, False, "Failed: Symbolic Arm"

        # Try basic check for if the target is in the workspace.
        # Maximum length of the arm is sum(sqrt(d_i^2 + a_i^2)), distance to target is norm(A_t)
        maximum_reach = 0
        for i in range(self.n):  # Add max length of each link
            maximum_reach = maximum_reach + sp.sqrt(self.dh[i][0] ** 2 + self.dh[i][2] ** 2)

        pt = target[0:3, 3]  # Find distance to target
        target_distance = sp.sqrt(pt[0] ** 2 + pt[1] ** 2 + pt[2] ** 2)

        if target_distance > maximum_reach and not force:
            print("WARNING: Target outside of reachable workspace!")
            return q, False, "Failed: Out of workspace"
        else:
            if target_distance > maximum_reach:
                print("Target out of workspace, but finding closest solution anyway")
            else:
                print("Target passes naive reach test, distance is {:.1} and max reach is {:.1}".format(
                    float(target_distance), float(maximum_reach)))

        if K is None:
            K = sp.eye(3) * 0.001
        if not isinstance(K, sp.Matrix):
            K = sp.eye(3) * K

        count = 0

        def get_error(q):
            c = self.fk(q)
            e = target[0:3, 3] - c[0:3, 3]
            return e

        def get_jacobian(q):
            J = self.jacob(q)
            return J[0:3, :]

        def get_jdag(J):
            Jdag = J.T @ (J @ J.T + sp.eye(3) * kd).inv()
            return Jdag

        e = get_error(q)

        qs = []

        ## Jacobian Transpose Method ##
        if method == 'jt':
            K_step = K
            while e.norm(2) > tol and count < max_iter:
                count += 1
                J = get_jacobian(q)
                qd = J.T @ K_step @ e
                qs.append(q)
                q = q + qd * dt
                e = get_error(q)
                sp.pprint(e.norm())
                print(count)

        if method == 'pinv':
            while e.norm() > tol and count < max_iter:
                count += 1
                J = get_jacobian(q)
                Jdag = get_jdag(J)
                qd = Jdag @ e
                q = q + qd * dt
                e = get_error(q)
                sp.pprint(e.norm())
                print(count)

        return (q, e, count, count < max_iter)

    def ik3(self, target, q0=None, tol=1e-6, max_iter=100):
        print(2)


def shifting_gamma(R, p):
    """"
    gamma = shifting_gamma(R, p)
    Description: 
    Returns the transform needed to shift a Jacobian from one point to a new point defined by the relative transform R and the translation p

    Parameters:
    R - 3x3 sympy matrix, the transform from the initial frame to the final frame, expressed in the initial frame (e.g. R^1_2)
    p - 3x1 sympy matrix length 3 iterable, the translation from the initial Jacobian point to the final point, expressed in the initial jacobian frame

    Returns:
    gamma - 6x6 sympy matrix, the transform gamma that shifts a jacobian
    """
    def skew(p):
        return sp.Matrix([[0, -p[0], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])
    S = skew(p)
    gamma = sp.Matrix([[R.T, -R.T @ S], [sp.zeros(3), R.T]])
    return gamma

def jacob_shift(J, R, p):
    """"
    J_shifted = jacob_shift(J, R, p)
    Description: 
    Shifts (rotates and translates) a Jacobian from one point to a new point defined by the relative transform R and the translation p

    Parameters:
    J - 6xN sympy matrix, represents the initial unshifted jacobian
    R - 3x3 sympy matrix, the transform from the initial frame to the final frame, expressed in the initial frame (e.g. R^1_2)
    p - 3x1 sympy matrix length 3 iterable, the translation from the initial Jacobian point to the final point, expressed in the initial jacobian frame

    Returns:
    J_shifted - 6xN sympy matrix, the new shifted jacobian
    """

    gamma = shifting_gamma(R, p)
    J_shifted = gamma @ J
    return J_shifted

if __name__ == "__main__":
    print(None)