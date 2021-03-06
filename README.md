## Multi-Task Learning for Simultaneous Speed-of-Sound Mapping and Image Reconstruction Using Non-Contact Thermoacoustics

### Multi-input, Multi-task Architecture
![Architecture](https://raw.githubusercontent.com/maxlwang/mtl-sos-recon/master/models/mtl_model2.png)

### Organization
The repository is organized as follows:

- dataset_generation contains 
  - binroots: binary masks of root images
  - k-Wave_sims: code for k-wave simulations    
  - Functions: functions for SoS map generation and image reconstruction
  - training_data_preprocessing: crop, normalize and save training data as images

- models contains
  - models.py: functions for implementing U-Net and multi-task U-Net models
  - baseline: jupyter notebook for vaniila U-Net implementation
  - simo: jupyter notebook for multi-task network with a-scan input
  - mimo: jupyter notebook for multi-input, multi-task network with a-scan + constant SoS image input

### Dependencies
Listed in the requirements.txt file
