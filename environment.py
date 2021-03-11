#Imports

import sys
vortex_folder = r'C:\CM Labs\Vortex Studio 2020b\bin'
sys.path.append(vortex_folder)

import Vortex as VxSim
import vxatp3 as vxatp
import numpy as np
from gym import spaces




#Environment Parameters
MAX_TORQUE = 1500
SUB_STEPS = 2

MAX_STEPS = 500
SPEED_PENALTY = 0.05
VERTICAL_REWARD = 2
HORIZONTAL_PENALTY = 0.35

X_AXIS_REWARD = 2
Y_AXIS_PENALTY = 1
SURVIVAL_REWARD = 0.1
ROTATION_PENALTY = 1
COLLISION_PENALTY = 1



class env():

    def __init__(self):

        self.vxmechanism = None
        self.max_speed = 8
        self.max_torque = 1

        self.setup_file = 'Vortex Resources\Setup.vxc'
        self.content_file = 'Vortex Assets\Dog.vxmechanism'

        self.application = vxatp.VxATPConfig.createApplication(self, 'Pendulum App', self.setup_file)

        #Set the Material Table
        materials = VxSim.VxExtensionFactory.create(VxSim.VxFactoryKey.createFromUuid('278b2748-90c2-5b5e-9aab-fa8c44bd4c82'))
        materials.getParameter("Material table file").setValue('Vortex Resources\default.vxmaterials')
        self.application.add(materials)

        #Initialize Action and Observation Spaces for the NN
        high = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.reward_range = (-200, 200)

        self.graphicsModule = VxSim.VxExtensionFactory.create(VxSim.GraphicsModuleICD.kModuleFactoryKey)
        self.application.insertModule(self.graphicsModule)

        # Create a display window
        # Create a display window
        self.display = VxSim.VxExtensionFactory.create(VxSim.DisplayICD.kExtensionFactoryKey)
        self.display.getInput(VxSim.DisplayICD.kPlacementMode).setValue("Windowed")
        self.display.setName('3D Display')
        self.display.getInput(VxSim.DisplayICD.kPlacement).setValue(VxSim.VxVector4(50, 50, 1280, 720))


    def reset(self):
        self.current_step = 0

        self.mechanism = None
        self.interface = None
        self.reward = 0

        # Switch to Editing
        vxatp.VxATPUtils.requestApplicationModeChangeAndWait(self.application, VxSim.kModeEditing)

        # The first time we load the mechanism
        if self.vxmechanism is None:
            # load mechanism file
            self.vxmechanism = self.application.getSimulationFileManager().loadObject(self.content_file)

            vxatp.VxATPUtils.requestApplicationModeChangeAndWait(self.application, VxSim.kModeSimulating)

            # Initialize first key frame
            self.application.update()
            self.keyFrameList = self.application.getContext().getKeyFrameManager().createKeyFrameList("KeyFrameList", False)
            self.application.update()

            self.keyFrameList.saveKeyFrame()
            self.waitForNbKeyFrames(1, self.application, self.keyFrameList)
            self.key_frames_array = self.keyFrameList.getKeyFrames()

        # Other times we reset the environment
        else:
            vxatp.VxATPUtils.requestApplicationModeChangeAndWait(self.application, VxSim.kModeSimulating)

            # Load first key frame
            self.keyFrameList.restore(self.key_frames_array[0])
            self.application.update()

        self.mechanism = VxSim.MechanismInterface(self.vxmechanism)
        self.interface = self.mechanism.findExtensionByName('RL_Interface')

        return self._get_obs()

    def step(self, actions): #takes a numpy array as input

        #Apply actions
        i = 0
        for action in actions:
            self.interface.getInputContainer()['input '+str(i)].value = float(action)
            i += 1

        #Step the simulation
        for i in range(SUB_STEPS):
            self.application.update()

        #Observations
        obs = self._get_obs()

        #rewards
        reward = 0
        if self.current_step > MAX_STEPS:
            done = True
        elif self.interface.getOutputContainer()[2].value:
            reward += - COLLISION_PENALTY
            done = False
        else:
            done = False
            reward += SURVIVAL_REWARD

        info = {}


        velocity = self.interface.getOutputContainer()[1].value

        reward += max(0,velocity.x) * X_AXIS_REWARD
        #reward += - transform.y * Y_AXIS_PENALTY
        #reward += - (abs(obs[1]) + abs(obs[2]) + abs(obs[3])) * ROTATION_PENALTY


        self.current_step += 1

        return obs, reward, done, info

    def _get_obs(self):

        #Extract values from RL_Interface
        transform = VxSim.getTranslation(self.interface.getOutputContainer()[0].value)
        rotation = VxSim.getRotation(self.interface.getOutputContainer()[0].value)
        velocity = self.interface.getOutputContainer()[1].value

        output = []
        for index, position in enumerate(self.interface.getOutputContainer()):
            if index >= 3:
                output.append(position.value)

        output.append(transform.z)

        output.append(rotation.x)
        output.append(rotation.y)
        output.append(rotation.z)

        output.append(velocity.x)
        output.append(velocity.y)
        output.append(velocity.z)

        return np.array(output)


    def render(self,  active=True, sync=False):
        #Find current list of displays
        current_displays = self.application.findExtensionsByName('3D Display')

        #If active, add a display and activate Vsync
        if active:
            if len(current_displays) == 0:
                self.application.add(self.display)

            if sync:
                self.application.setSyncMode(VxSim.kSyncSoftwareAndVSync)
            else:
                self.application.setSyncMode(VxSim.kSyncNone)

        #If not, remove the current display and deactivate Vsync
        else:
            if len(current_displays) == 1:
                self.application.remove(current_displays[0])
            self.application.setSyncMode(VxSim.kSyncNone)


    def waitForNbKeyFrames(self,expectedNbKeyFrames, application, keyFrameList):
        maxNbIter = 100
        nbIter = 0
        while len(keyFrameList.getKeyFrames()) != expectedNbKeyFrames and nbIter < maxNbIter:
            if not application.update():
                break
            ++nbIter

