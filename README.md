0. If you're using Nix on the OpenStack, use the provided flake.

```
nix develop
```

1. Build the project (in Debug or Release) with cmake

```
export builddir=... # pas dans l'AFS
cmake -S . -B $builddir -DCMAKE_BUILD_TYPE=Debug
```

or

```
cmake -S . -B $builddir -DCMAKE_BUILD_TYPE=Release
```

2. Compile with Make:

```
make -C $builddir
```

3. Run with

```
$builddir/stream --mode=[gpu,cpu] <video.mp4> [--output=output.mp4]
```

4. Edit your cuda/cpp code in */Compute.*
