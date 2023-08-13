# NORDIC_ATIC
The annual nordic conference in advanced topics in control(ATIC). This it the repository for our project for the course **Advanced Topics in Control** at ETH spring 2023. Check out the project website for more information and visualizations: https://larsrpe.github.io/NORDIC_ATIC/

# Requirements
To run this you will need pytorch,numpy,scipy and tqdm. You will also need matplotlib for visualizations.

# Demonstration
Running main.py will reproduce the results from the paper. To be able to run the walking man example you will need to download the mp4 video from this [link](link:https://drive.google.com/file/d/1ohfWxChmzC5f34ISOxV8MOEoUlAwDv7Q/view?usp=sharing) and place it at:
```
videos
└── man_walking.mp4
```

Note that he required interpolation is time consuming and not entirely stable. For convinience you can download the precomputed interpolation coeffs from here: and place the file at:
```
data
└── walking_man
    └── resolution64.pt
```
To run the demonstration run python main.py. This will write some visualizations to the sims/ folder.
