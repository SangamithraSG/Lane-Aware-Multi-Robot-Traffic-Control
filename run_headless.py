import multi_robot_sim as mrs
import sys

def mock_animate(self):
    print('[INFO] Starting headless loop...')
    for _ in range(300):
        self.tick_step()
        if not self.running:
            break
    self.print_summary()

mrs.Simulation._animate = mock_animate
mrs.Simulation.run = mock_animate

sim = mrs.Simulation()
sim.run()
