# Changelog

All notable changes to this project will be documented in this file.

## v2.0.0

This releases completely reworks py3dtiles command line and add new features.

The command line now uses subcommands syntax, in order to expose a single
entry point and support multiple commands. The existing commands 'export_tileset' and 'py3dtiles_info' became
'py3dtiles export' and 'py3dtiles info'.

### Changes

- relicensed as Apache 2.0.
- minimal python supported is now 3.7
- dependencies versions has been updated:
    - laspy should be at least 2.0
    - numpy at least 1.20.0
- Tile has been renamed to TileContent

### Features

New features were added, the main one being: py3dtiles can be used to convert
pointcloud las files to a 3dtiles tileset.

This is the purpose of the 'py3dtiles convert' command. It supports multicore
processor for faster processing, leveraging pyzmq, and the memory management has been carefully
implemented to support virtually unlimited points count.

Other features:

- read points from xyz files
- Documentation are now published at https://oslandia.gitlab.io/py3dtiles

### Fixes

* 53580ba Jeremy Gaillard       fix: use y-up orientation for glTF objects in export script  
* 65d6f67 Jeremy Gaillard       fix: proper bounding box size in export script  
* 3603b00 Augustin Trancart     fix: reliably select triangulation projection plane and orientation  
* fd2105a jailln                Fix gltf min and max value
