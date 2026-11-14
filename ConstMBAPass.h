#pragma once

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"

struct ConstMBAPass : llvm::PassInfoMixin<ConstMBAPass>
{
    llvm::PreservedAnalyses run(llvm::Module& m, llvm::ModuleAnalysisManager&);
};

bool RunConstMBA(llvm::Module& m);

extern llvm::cl::opt<bool> ConstMBAEnable;