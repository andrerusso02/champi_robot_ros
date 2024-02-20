# Coupe de France de Robotique 2024 : Code ROS2

## Requirements

- Ubuntu 22
- ROS2 Humble

## Commandements

Suis ces commandemments et tout se passera bien.

#### 1) Ne jamais créer de package python

Avec des packages c++, on peut faire aussi des nodes python. Mais avec des package python: 
- c'est super galère d'ajouter des nodes et des fichiers dans le projet
- bug avec symlink-install qui ne marche pas pour les launch files et les fichiers de config.
Regarder ce site pour savoir commment organiser un package c++ + python : https://roboticsbackend.com/ros2-package-for-both-python-and-cpp-nodes/

## How to

### Compile projects
```bash
colcon build --symlink-install
```
With `--symlink-install`, you can edit python files, launch files and config files without re-compiling.

### Generate protobuf files for the CAN Bus
```bash
cd non_ros/gen_proto
chmod +x gen.sh
./gen.sh
```

### U have an issue u don't understand :((
```bash
rm -R log build install # then recompile, and hope
```

## Nice ressources

* CAN cpp examples: https://github.com/craigpeacock/CAN-Examples/tree/master

## Notes

### setup.py deprecation warning

Pas de solution pour le moment : https://github.com/ament/ament_cmake/issues/382
