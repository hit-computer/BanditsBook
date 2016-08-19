import matplotlib.pyplot as plt
execfile("core.py")
#Initialize arms with mean and standard deviation
arm1 = NormalArm(0.1,2)
arm2 = NormalArm(0.2,2)
arm3 = NormalArm(0.3,2)
arm4 = NormalArm(0.4,2)
arm5 = NormalArm(0.5,2)
arm6 = NormalArm(0.6,2)
arm7 = NormalArm(0.7,2)
arm8 = NormalArm(0.8,2) #<------------Optimal Strategy
arms = [arm1, arm2, arm3, arm4, arm5, arm6, arm7, arm8]
n_arms = len(arms)
algo = EpsilonGreedy(0.05, [], [])
algo.initialize(n_arms)
avg_rewards = [0]
num_sims = 10000
rewards = []
arm_selections = []
change_of_distribution = False

for t in range(1,num_sims):
  #Change the underlying distribution of rewards
  if t == num_sims/2 and change_of_distribution:
    arm1 = NormalArm(0.8,2) #<------------Optimal Strategy
    arm2 = NormalArm(0.6,2)
    arm3 = NormalArm(0.7,2)
    arm4 = NormalArm(0.4,2)
    arm5 = NormalArm(0.3,2)
    arm6 = NormalArm(0.2,2)
    arm7 = NormalArm(0.1,2)
    arm8 = NormalArm(0.5,2)
    arms = [arm1, arm2, arm3, arm4, arm5, arm6, arm7, arm8]
    algo.initialize(n_arms)
  chosen_arm = algo.select_arm()
  arm_selections.append(chosen_arm+1)
  reward = arms[chosen_arm].draw()
  rewards.append(reward)
  running_avg = (avg_rewards[-1]*(t-1)+reward)/float(t)
  avg_rewards.append(running_avg)
  algo.update(chosen_arm, reward)

plt.figure()
plt.subplot(211)
plt.plot(avg_rewards, label='Average rewards')
if change_of_distribution:
  plt.axvline(x=num_sims/2, color='r')
plt.ylabel('Average Reward for Epsilon-Greedy')

plt.subplot(212)
plt.scatter(range(num_sims-1),arm_selections,  s=1)
plt.axis([0, num_sims, 0, 9])
if change_of_distribution:
  plt.axvline(x=num_sims/2, color='r')
plt.ylabel('Arm Selection')
plt.show()
