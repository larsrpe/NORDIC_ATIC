# NORDIC ATIC
Our research on time-varying density control of robotic swarms for the annual **Nordic Conference in Advance Topics in Control (NCATIC)** 

This it the repository for our project for the course **Advanced Topics in Control** at ETH spring 2023. Check out the [project website](https://larsrpe.github.io/NORDIC_ATIC/) for more information and visualizations.

# Requirements
To run this you will need pytorch,numpy,scipy and tqdm. You will also need matplotlib for visualizations.

# Demonstration
Running main.py will reproduce the results from the paper. To be able to run the walking man example you have to download the mp4 video from this [link](https://drive.google.com/file/d/1ohfWxChmzC5f34ISOxV8MOEoUlAwDv7Q/view?usp=sharing) and place it at:
```
videos
└── man_walking.mp4
```

Note that he required interpolation is time consuming and not entirely stable. For convinience you can download the precomputed interpolation coefficients from this [link](https://drive.google.com/file/d/1ohfWxChmzC5f34ISOxV8MOEoUlAwDv7Q/view?usp=sharing](https://drive.google.com/file/d/1MrFCiCpDs8wl264FPGL1QTCa88AL8O0v/view?usp=sharing) and place the file at:
```
data
└── walking_man
    └── resolution64.pt
```
To run the demonstration run python main.py. This will write some visualizations to the sims/ folder.

Note that for the **Nordic Conference in Advance Topics in Control** is purely fictional and only excists in the minds of the authors- for now, at least.
