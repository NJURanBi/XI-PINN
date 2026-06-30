# XI-PINN

Physics-informed neural networks (PINNs) have emerged as an effective class of mesh-free methods for solving partial differential equations (PDEs), particularly on complex geometries. In this paper, we introduce an Extended Interface Physics-Informed Neural Network (XI-PINN) framework designed to solve parabolic moving interface problems. The proposed method employs a level set function—which can be either analytically prescribed or learned via a neural network—to capture the moving interface. Furthermore, we establish an a priori error analysis for the XI-PINN method and derive error bounds for the approximation. \cb{Extensive numerical experiments are provided to validate the accuracy and robustness of the framework, and its applicability is further demonstrated by solving the Oseen equations.

<img width="3092" height="1024" alt="pred_error_crossection" src="https://github.com/user-attachments/assets/e681c164-6057-4749-8793-c036fa4d2ecb" />
<img width="3086" height="1024" alt="pred_error_crossection" src="https://github.com/user-attachments/assets/5bf9eb2d-b584-4c3c-ab0b-437280b871b4" />
