# The Rust Version of KrecViz

**NOTE:** This repo is temporary, I will move this logic to the proper [krecviz](https://github.com/kscalelabs/krecviz) package soon.

There are just two files, main.rs deals with the command line and running logic, and urdf_logger.rs deals with the urdf loading logic. 

Right now, all this does is loads a static urdf to the rerun viewer with all the 3D meshes.

![image](https://github.com/user-attachments/assets/e4da2fbc-df11-4924-bf74-c25f4a377888)


## Usage

Should just work with 
```bash
# clone this repo
# cd to the root of this repo
cargo run -- --urdf data/urdf_examples/gpr/robot.urdf
```
