# Polynomial Opaque Predicate Generator

Similar to the [Bitwise Opaque Generator](https://github.com/IntelSDM/BitwiseOpaquePredicateGenerator); Except we emit a known constant, the value doesn't matter; we then encode with a synthesised horners that is purposefully rigged to give certain value characteristics. <br>
This succeeds the bitwise method as the bitwise method can just be bruteforced, pass in any value, it will always be true/false, its equation itself is either impossible or always possible. <br>
Since we control the constant here, it changes this, as a different constant provides different values, then the runtime transform through the horners; this means you can't SMT the values without symbolic execution of the unique horners algorithm. <br>
Sure; most people symbolically execute, but its just an example of stepping up from the old school bitwise techniques. <br>

<img width="905" height="672" alt="image" src="https://github.com/user-attachments/assets/24a853c2-f802-458c-abe7-24a2f44d1c1f" /> <br>


## Annotations
```c
__attribute__((annotate("obfus")))
``` 
* Marks your function for the pass

## How To Build
* Download [LLVM 19.1](https://github.com/llvm/llvm-project/releases/tag/llvmorg-19.1.0)
* Download [VCPKG](https://github.com/microsoft/vcpkg)
* VCPKG install asmjit
* Open build.bat in a text editor.
* Find the LLVM 19.1 references in the command line, Change them to your file path.
* Find the VCPKG references in the command line, Change them to your file path.
* Run the build script
* Build your application with the ``` -load-pass-plugin ``` Param to the FunctionPass.dll
