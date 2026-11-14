#pragma once

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"

struct RiggedHornerOpaquePass : llvm::PassInfoMixin<RiggedHornerOpaquePass>
{
    llvm::PreservedAnalyses run(llvm::Module& m, llvm::ModuleAnalysisManager&);
};

bool RunRiggedHornerOpaque(llvm::Module& m);

extern llvm::cl::opt<bool> RiggedHornerOpaqueEnable;