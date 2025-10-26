EPISODES ?= 5
TRAIN_STEPS ?= 50000
EVAL_EPISODES ?= 5
SEED ?= 1


install_requirements:
	@pip install -r requirements.txt

run_sonic_emeraldHillZone_act1:
	python SonicTheHedgehog2/main_sonic.py --game SonicTheHedgehog2-Genesis --state ${GENESIS_GAME}  --episode 34 --sonic-helper

run_sonic_aquaticRuinZone_act1:
	python SonicTheHedgehog2/main_sonic.py --game SonicTheHedgehog2-Genesis --state AquaticRuinZone.Act1 --episode 34 --sonic-helper

run_sonic_aquaticRuinZone_act2:
	python SonicTheHedgehog2/main_sonic.py --game SonicTheHedgehog2-Genesis --state AquaticRuinZone.Act2 --episode 34 --sonic-helper

run_sonic_casinoNightZone_act1:
	python SonicTheHedgehog2/main_sonic.py --game SonicTheHedgehog2-Genesis --state CasinoNightZone.Act1 --episode 34 --sonic-helper

run_sonic_casinoNightZone_act2:
	python SonicTheHedgehog2/main_sonic.py --game SonicTheHedgehog2-Genesis --state CasinoNightZone.Act2 --episode 34 --sonic-helper

run_sonic_chemicalPlantZone_act1:
	python SonicTheHedgehog2/main_sonic.py --game SonicTheHedgehog2-Genesis --state ChemicalPlantZone.Act1 --episode 34 --sonic-helper

run_sonic_chemicalPlantZone_act2:
	python SonicTheHedgehog2/main_sonic.py --game SonicTheHedgehog2-Genesis --state ChemicalPlantZone.Act2 --episode 34 --sonic-helper

run_sonic_emeraldHillZone_act2:
	python SonicTheHedgehog2/main_sonic.py --game SonicTheHedgehog2-Genesis --state EmeraldHillZone.Act2 --episode 34 --sonic-helper

run_sonic_hillTopZone_act1:
	python SonicTheHedgehog2/main_sonic.py --game SonicTheHedgehog2-Genesis --state HillTopZone.Act1 --episode 34 --sonic-helper

run_sonic_hillTopZone_act2:
	python SonicTheHedgehog2/main_sonic.py --game SonicTheHedgehog2-Genesis --state HillTopZone.Act2 --episode 34 --sonic-helper

run_sonic_metropolisZone_act1:
	python SonicTheHedgehog2/main_sonic.py --game SonicTheHedgehog2-Genesis --state MetropolisZone.Act1 --episode 34 --sonic-helper

run_sonic_metropolisZone_act2:
	python SonicTheHedgehog2/main_sonic.py --game SonicTheHedgehog2-Genesis --state MetropolisZone.Act2 --episode 34 --sonic-helper

run_sonic_metropolisZone_act3:
	python SonicTheHedgehog2/main_sonic.py --game SonicTheHedgehog2-Genesis --state MetropolisZone.Act3 --episode 34 --sonic-helper

run_sonic_mysticCaveZone_act1:
	python SonicTheHedgehog2/main_sonic.py --game SonicTheHedgehog2-Genesis --state MysticCaveZone.Act1 --episode 34 --sonic-helper

run_sonic_mysticCaveZone_act2:
	python SonicTheHedgehog2/main_sonic.py --game SonicTheHedgehog2-Genesis --state MysticCaveZone.Act2 --episode 34 --sonic-helper

run_sonic_oilOceanZone_act1:
	python SonicTheHedgehog2/main_sonic.py --game SonicTheHedgehog2-Genesis --state OilOceanZone.Act1 --episode 34 --sonic-helper

run_sonic_oilOceanZone_act2:
	python SonicTheHedgehog2/main_sonic.py --game SonicTheHedgehog2-Genesis --state OilOceanZone.Act2 --episode 34 --sonic-helper

run_sonic_wingFortressZone:
	python SonicTheHedgehog2/main_sonic.py --game SonicTheHedgehog2-Genesis --state WingFortressZone --episode 34 --sonic-helper

run_lunar_lander:
	python LunarLander-project-main/main_lander.py --random --episodes 5 --render

run_lunar_random:
	python LunarLander-project-main/main_lunar.py --random --episodes $(EPISODES) $(RENDER)

run_lunar_train:
	python LunarLander-project-main/main_lunar.py --train --train_steps $(TRAIN_STEPS) --seed $(SEED)

run_lunar_eval:
	python LunarLander-project-main/main_lunar.py --train --train_steps $(TRAIN_STEPS) --eval_episodes $(EVAL_EPISODES) --seed $(SEED)

run_lunar_random_no_render:
	python LunarLander-project-main/main_lunar.py --random --episodes $(EPISODES)

run_lunar_custom_seed:
	python LunarLander-project-main/main_lunar.py --train --train_steps $(TRAIN_STEPS) --seed $(SEED)

run_lunar_full_cycle:
	python LunarLander-project-main/main_lunar.py --train --train_steps $(TRAIN_STEPS) --eval_episodes $(EVAL_EPISODES) $(RENDER) --seed $(SEED)

run_lunar_trained:
	python LunarLander-project-main/main_lunar.py --run_trained --eval_episodes $(EVAL_EPISODES) $(RENDER) --seed $(SEED) --save_video
retro_sonic:
	sudo apt-get install python3-opengl
	python -m retro.import SonicTheHedgehog2/Sonic\ The\ Hedgehog\ 2\ \(World\)\ \(Rev\ A\)
clean_lunar_cache:
	rm -rf LunarLander-project-main/__pycache__

delete_lunar_model:
	rm -rf LunarLander-project-main/dqn_lunarlander.pth