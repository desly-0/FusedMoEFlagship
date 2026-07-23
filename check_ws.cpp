#include <iostream>
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"

int main() {
    aclInit(nullptr);
    int32_t devId = 0;
    aclrtSetDevice(devId);

    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    size_t sysWsSize = platform->GetLibApiWorkSpaceSize();
    std::cout << "GetLibApiWorkSpaceSize() = " << sysWsSize << std::endl;

    uint32_t coreNum = platform->GetCoreNumAic();
    std::cout << "GetCoreNumAic() = " << coreNum << std::endl;

    aclrtResetDevice(devId);
    aclFinalize();
    return 0;
}
