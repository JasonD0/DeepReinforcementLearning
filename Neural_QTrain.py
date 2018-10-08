import gym
import tensorflow as tf
import numpy as np
import random

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA = 0.99 # discount factor
INITIAL_EPSILON = 0.9 # starting value of epsilon
FINAL_EPSILON = 0.1 # final value of epsilon
EPSILON_DECAY_STEPS = 1000 # decay period

# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])

# TODO: Define Network Graph
#w1 = tf.Variable(tf.random_normal([STATE_DIM, 10], stddev=0.05), dtype=tf.float32, name="weigthts1")  
#b1 = tf.Variable(tf.zeros([10]), dtype=tf.float32, name="biases1")
w1 = tf.get_variable(name="weights1", shape=[STATE_DIM, 10], dtype=tf.float32, initializer=tf.random_normal_initializer(0.1, 0.3))
b1 = tf.get_variable(name="biases1", shape=[10], dtype=tf.float32, initializer=tf.constant_initializer(0.1))

#h1 = tf.nn.relu(tf.matmul(state_in, w1) + b1,  name="hidden1")
h1 = tf.nn.relu(tf.matmul(state_in, w1) + b1)

#w2 = tf.Variable(tf.random_normal([10, 1], stddev=0.05), dtype=tf.float32, name="weights2")
#b2 = tf.Variable(tf.zeros([1]), dtype=tf.float32, name="biases2")
w2 = tf.get_variable(name="weights2", shape=[10, ACTION_DIM], dtype=tf.float32, initializer=tf.random_normal_initializer(0.1, 0.3))
b2 = tf.get_variable(name="biases2", shape=[ACTION_DIM], dtype=tf.float32, initializer=tf.constant_initializer(0.1))

# TODO: Network outputs
q_values = tf.matmul(h1, w2) + b2;
q_action = tf.reduce_sum(tf.multiply(q_values, action_in)) # action_in 

# TODO: Loss/Optimizer Definition
#loss = tf.reduce_mean(-tf.reduce_sum(target_in * tf.log(q_action + 1e-2)), name="loss")
loss = tf.reduce_mean(tf.squared_difference(target_in, q_action))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
#optimizer = tf.train.AdamOptimizer().minimize(loss)

# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action

# Main learning loop
for episode in range(EPISODE):

    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    epsilon -= epsilon / EPSILON_DECAY_STEPS

    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))

        nextstate_q_values = q_values.eval(feed_dict={
            state_in: [next_state]
        })

        # TODO: Calculate the target q-value.
        # hint1: Bellman
        # hint2: consider if the episode has terminated
        target = reward + GAMMA*np.max(nextstate_q_values)*(1.0 - done) 

        # Do one training step
        session.run([optimizer], feed_dict={
            target_in: [target],
            action_in: [action],
            state_in: [state]
        })

        # Update
        state = next_state
        if done:
            break

    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()