
# NOTE: the agent holds items as a list where
# items[idx] is the number of items collected of type 'idx'

# IMPORTANT: if the original item config is changed/reodered, this file will
# likely have to be updated to reflect new item positions.

def all_item_reward(ai_r=1):
  # awards +ai_r for every new item collected
  def reward_calc(prev_items, items):
    # hardcoded based on config specification
    return (sum(items)-sum(prev_items))*ai_r

  return reward_calc

def r_jellybean(jb_r=1):
  # rewards +jb_r for every jellybean collected
  def reward_calc(prev_items, items):
    # hardcoded based on config specification
    JB_IDX = 2
    return (items[JB_IDX]-prev_items[JB_IDX])*jb_r

  return reward_calc

def r_onion(on_r=1):
  # rewards +on_r for every onion collected
  def reward_calc(prev_items, items):
    # hardcoded based on config specification
    ON_IDX = 1
    return (items[ON_IDX]-prev_items[ON_IDX])*on_r

  return reward_calc

def r_banana(ban_r=1):
  # rewards +ban_r for every banana collected
  def reward_calc(prev_items, items):
    # hardcoded based on config specification
    BAN_IDX = 0
    return (items[BAN_IDX]-prev_items[BAN_IDX])*ban_r

  return reward_calc

def avoid_onion():
  LIVING_REWARD = -0.005
  # LIVING_REWARD = -0.00
  ONION_PENALTY = -1
  SUCCESS_REWARD = +10

  def reward_calc(prev_items, items):
    if sum(prev_items)==sum(items):
      # implies no items were found
      return LIVING_REWARD

    onion_penalty = r_onion(ONION_PENALTY)(prev_items, items)

    if onion_penalty == 0:
      # implies item picked up wasn't an onion
      return all_item_reward(SUCCESS_REWARD)(prev_items, items)

    return onion_penalty

  return reward_calc

