package main

import (
	"log"
	"os"

	mqtt "github.com/eclipse/paho.mqtt.golang"
	"gopkg.in/yaml.v2"
)

type MqttConfig struct {
	Mqtt struct {
		BrokerAddress string `yaml:"broker_address"`
		ClientID      string `yaml:"client_id"`
		Username      string `yaml:"username"`
		Password      string `yaml:"password"`
		Topics        struct {
			Rtt string `yaml:"rtt"`
		} `yaml:"topics"`
	} `yaml:"mqtt"`
}

type MqttClient struct {
	Client mqtt.Client
	Cfg    MqttConfig
}

type MqttPayload struct {
	Timestamp string      `json:"timestamp"`
	ID        string      `json:"id"`
	Msg       interface{} `json:"msg"`
}

func (mc *MqttClient) Init(filename string) error {
	var cfg MqttConfig

	f, err := os.ReadFile(filename)
	if err != nil {
		return err
	}

	if err := yaml.Unmarshal(f, &cfg); err != nil {
		return err
	}

	mc.Cfg = cfg

	opts := mqtt.NewClientOptions()
	opts.AddBroker(cfg.Mqtt.BrokerAddress)
	opts.SetClientID(cfg.Mqtt.ClientID)
	opts.SetUsername(cfg.Mqtt.Username)
	opts.SetPassword(cfg.Mqtt.Password)

	mc.Client = mqtt.NewClient(opts)
	if token := mc.Client.Connect(); token.Wait() && token.Error() != nil {
		return token.Error()
	}

	return nil
}

func (mc *MqttClient) Close() {
	mc.Client.Disconnect(250)
}

func (mc *MqttClient) Publish(topic string, payload interface{}, qos byte, retained bool) {
	if token := mc.Client.Publish(topic, qos, retained, payload); token.Wait() && token.Error() != nil {
		log.Panicf("ERROR: failed to publish message: %s", token.Error())
	}
}

// add subscribe with a callback function if needed
