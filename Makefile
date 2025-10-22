
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

retro_sonic:
	sudo apt-get install python3-opengl
	python -m retro.import SonicTheHedgehog2/Sonic\ The\ Hedgehog\ 2\ \(World\)\ \(Rev\ A\)
