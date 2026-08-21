import numpy as np

################################################################################################################################################################
# Environment classes

# Resource bins:
#    full      high       med       low      none
# [100, 85], (85, 65], (65, 35], (35, 15], (15, 0]
#     15        20        30        20        15

class Patch:
    def __init__(self, richness, resources, depletion_rate):
        # Patch qualities
        self.type = richness
        self.depletion_rate = depletion_rate
        self.initial_resources = resources # 0-4

        # Resource bins
        self.res_bins = np.array([0, 15, 35, 65, 85, 100])
        self.bin_sizes = np.array([15, 20, 30, 20, 15])

        # Initialise resources based on initial bin + upward noise
        res_idx = 4-resources # 0-4
        eta = np.random.uniform() # [0,1)
        self.resources = self.res_bins[res_idx] + eta * self.bin_sizes[res_idx] # get random initial value from correct bin

    def forage(self):
        # Forage resources from patch
        # Food consumption slows over time with some variance > concave curve to match MVT

        # Add noise to depletion rate
        noise = np.random.beta(10, 10) + 0.5 # symmetrical gaussian shape, bounded to (0.5, 1.5)
        depletion = noise * self.depletion_rate

        # Exponential decay
        res = self.resources * (1.0 - depletion)
        harvest = self.resources - res
        self.resources = res

        # Convert number to discrete bins
        res_cat = 5 - np.digitize(self.resources, self.res_bins)

        #print(f"Resources on patch = {self.resources:2d} | Bin = {res_cat:1d}")
        return res_cat, harvest
        
class PatchWorld:
    def __init__(self, 
        A_matrix,
        travel_time=5, 
        richness_dist=np.array([    # {rich, medium, sparse} (patch type)
            0.3, 
            0.4,
            0.3
        ]),    
        resource_dist=np.array([    # {rich, medium, sparse} -> {full, high, med, low, none} (initial resources)
            [0.40, 0.00, 0.00],
            [0.50, 0.20, 0.00],
            [0.10, 0.60, 0.20],
            [0.00, 0.20, 0.60],
            [0.00, 0.00, 0.20]
        ]), 
        depletion_rate=0.20         # -20% (avg) on each forage step
        ):

        # Global patch info
        self.richness_dist = richness_dist
        self.resource_dist = resource_dist
        self.current_patch = None
        self.arrival_cue = None
        self.harvest = None
        self.depletion_rate = depletion_rate

        # Travel info
        self.travel_time = travel_time

        # Observation likelihood info
        self.A = A_matrix

    def generate_patch(self):
        # Reset patch
        self.current_patch = None

        # Sample from richness distribution
        richness = np.random.choice(3, p=self.richness_dist)

        # Sample from initial resource distribution
        resources = np.random.choice(5, p=self.resource_dist[:, richness])

        # Get depletion rate
        depletion_rate = self.depletion_rate

        # Construct patch object
        self.current_patch = Patch(richness, resources, depletion_rate)

        # Sample from A[0] matrix to get arrival cue (constant per patch)
        self.arrival_cue = np.random.choice(self.A[0].shape[0], p=self.A[0][:, self.current_patch.type])

        # return self.current_patch

    def travel(self, travel_step):
        return [3, 3, travel_step] # [travel, no_food, travel_i]
 
    def forage(self):
        # Forage from patch to get resource amount
        resources, self.harvest = self.current_patch.forage() # {full, high, med, low, none} = 0-4
        
        # Sample from A[1] matrix to get food observation
        food_obs = np.random.choice(self.A[1].shape[0], p=self.A[1][:, resources])
        #print(f"Food obs: {food_obs:1d}")
        return [self.arrival_cue, food_obs, 0] # 0 = on_patch

    def last_harvest(self):
        return self.harvest