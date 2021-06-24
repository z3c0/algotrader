
from components import (AccountComponent, OptimizerComponent,
                        RiskManagementComponent, TraderComponent)


class Strategy:
    account = AccountComponent()
    optimizer = OptimizerComponent()
    risk = RiskManagementComponent()
    trader = TraderComponent()
