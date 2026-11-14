#pragma once

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"

struct LinearMBAPass : llvm::PassInfoMixin<LinearMBAPass>
{
    llvm::PreservedAnalyses run(llvm::Module& m, llvm::ModuleAnalysisManager&);
};

bool RunLinearMBA(llvm::Module& m);

extern llvm::cl::opt<bool> LinearMBAEnable;