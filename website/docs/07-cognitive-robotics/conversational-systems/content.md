---
title: Conversational Systems for Humanoid Robots
sidebar_position: 1
---

# Conversational Systems for Humanoid Robots

## Introduction

Conversational systems form the foundation of natural human-robot interaction, enabling humanoid robots to communicate with humans using natural language. This capability is essential for humanoid robots operating in human environments, as it allows them to receive instructions, provide information, and engage in meaningful social interactions. This chapter explores the design, implementation, and evaluation of conversational systems specifically tailored for humanoid robotics applications.

## Fundamentals of Conversational AI for Robotics

### Components of Conversational Systems

A complete conversational system for humanoid robots involves multiple interconnected components:

1. **Automatic Speech Recognition (ASR)**: Converting speech to text
2. **Natural Language Understanding (NLU)**: Interpreting the meaning of text
3. **Dialog Management**: Maintaining conversation context and flow
4. **Natural Language Generation (NLG)**: Producing appropriate responses
5. **Text-to-Speech (TTS)**: Converting text responses to speech
6. **Multimodal Integration**: Coordinating with gestures, facial expressions, and actions

```python
import numpy as np
import speech_recognition as sr
import pyttsx3
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class ConversationState:
    """Represents the state of an ongoing conversation"""
    current_topic: str = "greeting"
    context_memory: Dict[str, Any] = None
    last_spoken_time: float = 0.0
    user_intent: str = "unknown"
    user_preferences: Dict[str, Any] = None
    conversation_history: List[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.context_memory is None:
            self.context_memory = {}
        if self.user_preferences is None:
            self.user_preferences = {}
        if self.conversation_history is None:
            self.conversation_history = []

class SpeechRecognizer:
    """Handles speech recognition for the conversational system"""
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
    
    def listen_and_recognize(self, timeout: float = 5.0) -> Optional[str]:
        """Listen to user speech and return recognized text"""
        try:
            with self.microphone as source:
                print("Listening...")
                audio = self.recognizer.listen(source, timeout=timeout)
            
            # Using Google's speech recognition (requires internet)
            text = self.recognizer.recognize_google(audio)
            print(f"Recognized: {text}")
            return text
        
        except sr.WaitTimeoutError:
            print("Timeout: No speech detected")
            return None
        except sr.UnknownValueError:
            print("Speech was not understood")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None

class IntentClassifier:
    """Classifies user intents from recognized speech"""
    def __init__(self):
        # Define possible intents and keywords
        self.intents = {
            "greeting": ["hello", "hi", "hey", "good morning", "good evening", "how are you"],
            "navigation": ["go to", "move to", "walk to", "navigate to", "go", "move", "walk", "move forward"],
            "manipulation": ["pick up", "grab", "take", "hold", "lift", "get", "carry", "place", "put down"],
            "information_request": ["what", "where", "who", "when", "how", "can you tell me", "do you know"],
            "social_interaction": ["chat", "talk", "converse", "make conversation", "tell me a story"],
            "farewell": ["goodbye", "bye", "good night", "see you", "cya", "farewell"],
            "help": ["help", "assist", "what can you do", "how to", "instructions", "what are you capable of"]
        }
    
    def classify_intent(self, text: str) -> str:
        """Classify the intent of the given text"""
        text_lower = text.lower()
        
        # Check for each intent's keywords
        for intent, keywords in self.intents.items():
            if any(keyword in text_lower for keyword in keywords):
                return intent
        
        # If no intent is clearly identified, return "unknown"
        return "unknown"
    
    def extract_entities(self, text: str) -> Dict[str, str]:
        """Extract named entities from the text"""
        entities = {}
        
        # Simple entity extraction patterns
        # In a real implementation, use NER or more sophisticated methods
        import re
        
        # Extract potential locations
        location_patterns = [
            r"to the (\w+)",  # "go to the kitchen"
            r"(\w+) room",    # "kitchen room"
            r"(\w+) area"     # "dining area"
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                entities["location"] = matches[0]
                break
        
        # Extract object names
        object_patterns = [
            r"pick up the (\w+)",   # "pick up the cup"
            r"grab the (\w+)",     # "grab the book"
            r"take the (\w+)"      # "take the box"
        ]
        
        for pattern in object_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                entities["object"] = matches[0]
                break
        
        return entities

class DialogManager:
    """Manages dialog flow and conversation context"""
    def __init__(self):
        self.conversation_state = ConversationState()
        self.response_templates = {
            "greeting": [
                "Hello! How can I assist you today?",
                "Hi there! What would you like to do?",
                "Good day! How may I help you?"
            ],
            "navigation": [
                "I can help you with navigation. Where would you like to go?",
                "I'm ready to navigate. Please specify a destination.",
                "I can move to various locations. Where do you want me to go?"
            ],
            "manipulation": [
                "I can manipulate objects in the environment. What would you like me to do?",
                "I'm capable of picking up and placing objects. What do you need?",
                "I can assist with object manipulation. Please specify what to do."
            ],
            "information_request": [
                "I can provide information about my capabilities, the environment, or other topics.",
                "I'm knowledgeable about various subjects. What would you like to know?",
                "I can answer questions about robots, technology, or general topics."
            ],
            "social_interaction": [
                "I enjoy social interaction! What would you like to chat about?",
                "I'm here to have a conversation. What interests you?",
                "Let's talk! What's on your mind?"
            ],
            "farewell": [
                "Goodbye! Feel free to call me if you need assistance.",
                "See you later! It was nice talking with you.",
                "Farewell! Have a great day!"
            ],
            "help": [
                "I can help with navigation, object manipulation, and general conversation.",
                "I'm capable of moving around, picking up objects, and chatting with you.",
                "I can assist with various tasks. Just tell me what you need!"
            ],
            "unknown": [
                "I'm not sure I understood. Could you please rephrase?",
                "Could you repeat that? I didn't catch that clearly.",
                "I'm sorry, I don't know how to help with that. Can I assist with something else?"
            ]
        }
    
    def update_conversation_state(self, user_input: str, intent: str, 
                                 entities: Dict[str, str]) -> None:
        """Update the conversation state based on user input"""
        # Update the last spoken time
        self.conversation_state.last_spoken_time = time.time()
        
        # Update user intent
        self.conversation_state.user_intent = intent
        
        # Update context memory with new entities
        self.conversation_state.context_memory.update(entities)
        
        # Add to conversation history
        self.conversation_state.conversation_history.append({
            "user_input": user_input,
            "intent": intent,
            "entities": entities,
            "timestamp": time.time()
        })
    
    def generate_response(self, intent: str) -> str:
        """Generate an appropriate response based on intent"""
        import random
        
        if intent in self.response_templates:
            responses = self.response_templates[intent]
            return random.choice(responses)
        else:
            return "I'm not sure how to respond to that."
    
    def manage_context(self) -> Dict[str, Any]:
        """Return current context for response generation"""
        return {
            "current_topic": self.conversation_state.current_topic,
            "user_intent": self.conversation_state.user_intent,
            "context_memory": self.conversation_state.context_memory,
            "user_preferences": self.conversation_state.user_preferences,
            "conversation_history": self.conversation_state.conversation_history[-5:]  # Last 5 exchanges
        }

class TextToSpeechSynthesizer:
    """Handles text-to-speech conversion for the robot"""
    def __init__(self):
        self.engine = pyttsx3.init()
        
        # Configure voice properties
        voices = self.engine.getProperty('voices')
        if voices:
            self.engine.setProperty('voice', voices[0].id)  # Use first available voice
        
        self.engine.setProperty('rate', 150)  # Words per minute
        self.engine.setProperty('volume', 0.8)  # Volume level (0.0 to 1.0)
    
    def speak(self, text: str) -> None:
        """Convert text to speech and play it"""
        print(f"Robot says: {text}")
        self.engine.say(text)
        self.engine.runAndWait()
    
    def set_voice_properties(self, rate: int = None, volume: float = None) -> None:
        """Adjust voice properties"""
        if rate is not None:
            self.engine.setProperty('rate', rate)
        if volume is not None:
            self.engine.setProperty('volume', volume)

class SocialBehaviorGenerator:
    """Generates appropriate social behaviors to accompany speech"""
    def __init__(self):
        self.behavior_templates = {
            "greeting": ["nod_head", "maintain_eye_contact", "smile"],
            "navigation": ["show_attention", "acknowledge_understanding", "head_turn_to_direction"],
            "manipulation": ["orient_torso", "eye_contact_with_object", "acknowledge_task"],
            "information_request": ["lean_forward_slightly", "maintain_eye_contact", "show_engagement"],
            "social_interaction": ["maintain_eye_contact", "use_hand_gestures", "show_emotions"],
            "farewell": ["wave", "nod_head", "maintain_eye_contact"],
            "help": ["show_attention", "acknowledge_request", "orient_posture_to_user"]
        }
    
    def get_social_behaviors(self, intent: str) -> List[str]:
        """Get appropriate social behaviors for the given intent"""
        if intent in self.behavior_templates:
            return self.behavior_templates[intent]
        else:
            # Default behaviors for unknown intents
            return ["maintain_eye_contact", "show_attention"]

class ConversationalRobotSystem:
    """Main system integrating all conversational components"""
    def __init__(self):
        self.speech_recognizer = SpeechRecognizer()
        self.intent_classifier = IntentClassifier()
        self.dialog_manager = DialogManager()
        self.text_to_speech = TextToSpeechSynthesizer()
        self.social_behavior_gen = SocialBehaviorGenerator()
        
        self.is_active = True
        self.conversation_active = False
    
    def process_user_input(self, user_text: Optional[str] = None) -> Dict[str, Any]:
        """Process user input and generate response"""
        if user_text is None:
            # Listen to user speech
            user_text = self.speech_recognizer.listen_and_recognize()
        
        if user_text is None:
            return {
                "success": False,
                "message": "Could not understand user input",
                "response": "",
                "behavior": []
            }
        
        # Classify intent
        intent = self.intent_classifier.classify_intent(user_text)
        
        # Extract entities
        entities = self.intent_classifier.extract_entities(user_text)
        
        # Update dialog state
        self.dialog_manager.update_conversation_state(user_text, intent, entities)
        
        # Generate response
        response = self.dialog_manager.generate_response(intent)
        
        # Get appropriate social behaviors
        behaviors = self.social_behavior_gen.get_social_behaviors(intent)
        
        # Prepare result
        result = {
            "success": True,
            "message": user_text,
            "intent": intent,
            "entities": entities,
            "response": response,
            "behavior": behaviors
        }
        
        return result
    
    def run_conversation_cycle(self) -> bool:
        """Run one complete cycle of the conversation system"""
        if not self.is_active:
            return False
        
        try:
            # Listen to user
            result = self.process_user_input()
            
            if result["success"]:
                # Speak the response
                self.text_to_speech.speak(result["response"])
                
                # Execute social behaviors (in a real robot, this would trigger actuators)
                print(f"Social behaviors: {result['behavior']}")
                
                return True
            else:
                print("No input detected, continuing...")
                return True
                
        except KeyboardInterrupt:
            print("Conversation interrupted by user")
            return False
        except Exception as e:
            print(f"Error in conversation cycle: {e}")
            return True  # Continue running despite errors
    
    def start_conversation(self):
        """Start the main conversation loop"""
        print("Starting humanoid robot conversational system...")
        self.conversation_active = True
        
        while self.conversation_active:
            if not self.run_conversation_cycle():
                break
            time.sleep(0.1)  # Small delay to prevent excessive CPU usage
    
    def stop_conversation(self):
        """Stop the conversation system"""
        self.conversation_active = False
        print("Conversation system stopped")

# Example usage
if __name__ == "__main__":
    # Initialize the conversational system
    robot_conversation = ConversationalRobotSystem()
    
    # Simulate a few conversation exchanges
    test_inputs = [
        "Hello there!",
        "Can you please go to the kitchen?",
        "What's your name?",
        "Please pick up the red cup",
        "Goodbye!"
    ]
    
    print("Simulating conversation:")
    for i, user_input in enumerate(test_inputs):
        print(f"\nRound {i+1}: User says '{user_input}'")
        
        result = robot_conversation.process_user_input(user_input)
        
        print(f"Intent: {result['intent']}")
        print(f"Entities: {result['entities']}")
        print(f"Robot responds: '{result['response']}'")
        print(f"Social behaviors: {result['behavior']}")
```

## Natural Language Understanding for Robotics

### Domain-Specific Language Processing

Humanoid robots need to understand language in the context of their physical capabilities and environment:

```python
import spacy
import numpy as np
from typing import Dict, List, Any, Optional
import re

class RobotLanguageProcessor:
    """Processes natural language in the context of robot capabilities"""
    def __init__(self):
        # Load spaCy English model (this would need to be installed separately)
        # For this example, we'll implement simple NLP without external dependencies
        self.robot_actions = [
            "move", "go", "walk", "navigate", "turn", "rotate",
            "pick up", "grab", "take", "lift", "hold", "place", "put", "release",
            "greet", "wave", "nod", "smile", "speak", "talk",
            "find", "locate", "search", "look for",
            "follow", "come to", "approach", "get"
        ]
        
        # Object categories the robot can work with
        self.known_objects = [
            "cup", "bottle", "book", "phone", "pen", "keys", "wallet",
            "chair", "table", "couch", "sofa", "bed", "lamp", "box",
            "person", "human", "man", "woman", "child", "adult"
        ]
        
        # Location categories
        self.known_locations = [
            "kitchen", "living room", "bedroom", "bathroom", "office",
            "hallway", "dining room", "garden", "front door", "back door"
        ]
    
    def parse_robot_command(self, text: str) -> Dict[str, Any]:
        """Parse a natural language command for robot execution"""
        original_text = text
        text_lower = text.lower()
        
        # Initialize result structure
        result = {
            "action": "unknown",
            "object": None,
            "location": None,
            "direction": None,
            "distance": None,
            "person": None,
            "confidence": 0.0,
            "parsed_command": ""
        }
        
        # Extract action
        action = self._extract_action(text_lower)
        result["action"] = action
        
        # Extract object
        obj = self._extract_object(text_lower)
        result["object"] = obj
        
        # Extract location
        location = self._extract_location(text_lower)
        result["location"] = location
        
        # Extract direction
        direction = self._extract_direction(text_lower)
        result["direction"] = direction
        
        # Extract distance
        distance = self._extract_distance(text_lower)
        result["distance"] = distance
        
        # Extract person
        person = self._extract_person(text_lower)
        result["person"] = person
        
        # Calculate confidence based on how many elements were found
        confidence = 0.3  # Base confidence
        if action != "unknown": confidence += 0.2
        if obj: confidence += 0.15
        if location: confidence += 0.15
        if direction or distance: confidence += 0.1
        if person: confidence += 0.1
        
        result["confidence"] = min(confidence, 1.0)
        
        # Create a structured command
        result["parsed_command"] = self._create_structured_command(result)
        
        return result
    
    def _extract_action(self, text: str) -> str:
        """Extract action from text"""
        # Look for known actions, ordered by length to match longer phrases first
        sorted_actions = sorted(self.robot_actions, key=len, reverse=True)
        
        for action in sorted_actions:
            if action in text:
                return action
        
        return "unknown"
    
    def _extract_object(self, text: str) -> Optional[str]:
        """Extract object from text"""
        for obj in self.known_objects:
            # Check for the object with determiners
            patterns = [
                rf"\bthe\s+{obj}\b",
                rf"\ba\s+{obj}\b",
                rf"\ban\s+{obj}\b",
                rf"\b{obj}\b"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return obj
        
        return None
    
    def _extract_location(self, text: str) -> Optional[str]:
        """Extract location from text"""
        for loc in self.known_locations:
            # Check for the location with prepositions
            patterns = [
                rf"\bto\s+the\s+{loc}\b",
                rf"\bin\s+the\s+{loc}\b",
                rf"\bat\s+the\s+{loc}\b",
                rf"\b{loc}\b"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return loc
        
        return None
    
    def _extract_direction(self, text: str) -> Optional[str]:
        """Extract directional information from text"""
        directions = ["forward", "backward", "left", "right", "up", "down"]
        
        for dir_word in directions:
            if f"move {dir_word}" in text or f"go {dir_word}" in text or f"turn {dir_word}" in text:
                return dir_word
        
        # Check for relative directions
        if "north" in text or "south" in text or "east" in text or "west" in text:
            for dir_word in ["north", "south", "east", "west"]:
                if dir_word in text:
                    return dir_word
        
        return None
    
    def _extract_distance(self, text: str) -> Optional[float]:
        """Extract distance from text"""
        # Look for distance patterns like "3 meters", "2 feet", "10 cm", etc.
        distance_match = re.search(r'(\d+(?:\.\d+)?)\s*(meters|meter|m|feet|foot|ft|cm|centimeters|inches)', text)
        
        if distance_match:
            value, unit = distance_match.groups()
            value = float(value)
            
            # Convert to meters
            if unit in ['meters', 'meter', 'm']:
                return value
            elif unit in ['feet', 'foot', 'ft']:
                return value * 0.3048  # feet to meters
            elif unit == 'cm':
                return value / 100.0
            elif unit == 'centimeters':
                return value / 100.0
            elif unit == 'inches':
                return value * 0.0254
        
        # Look for relative distances
        if "a little" in text or "bit" in text:
            return 0.3  # 30cm
        elif "some" in text or "far" in text:
            return 1.0  # 1 meter
        elif "very" in text and "far" in text:
            return 2.0  # 2 meters
        
        return None
    
    def _extract_person(self, text: str) -> Optional[str]:
        """Extract person reference from text"""
        patterns = [
            r"\bto\s+(me|you|him|her|them)\b",
            r"\bthe\s+(man|woman|person|human)\b",
            r"\b(someone|anyone)\b"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return None
    
    def _create_structured_command(self, parsed_result: Dict[str, Any]) -> str:
        """Create a structured command from parsed elements"""
        command_parts = []
        
        action = parsed_result["action"]
        if action != "unknown":
            command_parts.append(action)
        
        obj = parsed_result["object"]
        if obj:
            command_parts.append(f"object: {obj}")
        
        location = parsed_result["location"]
        if location:
            command_parts.append(f"location: {location}")
        
        direction = parsed_result["direction"]
        if direction:
            command_parts.append(f"direction: {direction}")
        
        person = parsed_result["person"]
        if person:
            command_parts.append(f"person: {person}")
        
        distance = parsed_result["distance"]
        if distance is not None:
            command_parts.append(f"distance: {distance:.2f}m")
        
        return ", ".join(command_parts)

class RobotCommandExecutor:
    """Executes parsed commands on the robot"""
    def __init__(self):
        self.robot_position = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.is_holding_object = False
        self.held_object = None
        self.current_location = "starting_position"
        
        # Map of location coordinates (in a real system, these would come from localization)
        self.location_map = {
            "kitchen": np.array([2.0, 1.0, 0.0]),
            "living room": np.array([0.0, 2.0, 0.0]),
            "bedroom": np.array([3.0, 3.0, 0.0]),
            "office": np.array([-1.0, 1.0, 0.0])
        }
    
    def execute_command(self, parsed_command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the parsed command on the robot"""
        action = parsed_command["action"]
        location = parsed_command["location"]
        obj = parsed_command["object"]
        direction = parsed_command["direction"]
        distance = parsed_command["distance"]
        confidence = parsed_command["confidence"]
        
        result = {
            "success": False,
            "message": "",
            "robot_state": {
                "position": self.robot_position.copy(),
                "is_holding_object": self.is_holding_object,
                "held_object": self.held_object,
                "current_location": self.current_location
            }
        }
        
        # Only execute if confidence is high enough
        if confidence < 0.5:
            result["message"] = f"Command not understood with high confidence ({confidence:.2f}). Please rephrase."
            return result
        
        if action == "move" or action == "go" or action == "navigate":
            if location:
                # Navigate to specific location
                if location in self.location_map:
                    target_pos = self.location_map[location]
                    self._navigate_to(target_pos)
                    self.current_location = location
                    result["success"] = True
                    result["message"] = f"Navigated to {location}"
                else:
                    result["message"] = f"Unknown location: {location}"
            elif direction and distance:
                # Move in specific direction
                self._move_in_direction(direction, distance)
                result["success"] = True
                result["message"] = f"Moved {distance:.2f}m {direction}"
            else:
                result["message"] = "Please specify a destination or direction and distance"
        
        elif action == "pick up" or action == "grab" or action == "take":
            if obj:
                if not self.is_holding_object:
                    self.is_holding_object = True
                    self.held_object = obj
                    result["success"] = True
                    result["message"] = f"Picked up {obj}"
                else:
                    result["message"] = f"Already holding {self.held_object}. Please place it first."
            else:
                result["message"] = "Please specify which object to pick up"
        
        elif action == "place" or action == "put" or action == "release":
            if self.is_holding_object:
                self.is_holding_object = False
                prev_object = self.held_object
                self.held_object = None
                result["success"] = True
                result["message"] = f"Placed {prev_object}"
            else:
                result["message"] = "Not holding any object to place"
        
        elif action == "greet" or action == "wave":
            result["success"] = True
            result["message"] = "Waving to greet"
            # In a real robot, this would trigger waving motion
        
        else:
            result["message"] = f"Action '{action}' not implemented or not understood"
        
        return result
    
    def _navigate_to(self, target_pos: np.ndarray) -> None:
        """Navigate the robot to the target position"""
        # Simplified navigation - in reality, this would use path planning and control
        self.robot_position[:2] = target_pos[:2]  # Update x, y position
        print(f"Robot navigated to position: {self.robot_position[:2]}")
    
    def _move_in_direction(self, direction: str, distance: float) -> None:
        """Move the robot in the specified direction"""
        direction_map = {
            "forward": np.array([1, 0]),
            "backward": np.array([-1, 0]),
            "left": np.array([0, 1]),
            "right": np.array([0, -1])
        }
        
        if direction in direction_map:
            move_vector = direction_map[direction] * distance
            self.robot_position[:2] += move_vector
            print(f"Robot moved {distance}m {direction} to: {self.robot_position[:2]}")

# Example usage
language_processor = RobotLanguageProcessor()
command_executor = RobotCommandExecutor()

# Test commands
test_commands = [
    "Please go to the kitchen",
    "Can you grab the red cup?",
    "Move forward 2 meters",
    "Put the cup on the table",
    "Greet the person"
]

print("Testing robot command parsing and execution:")
for cmd in test_commands:
    print(f"\nInput: '{cmd}'")
    
    # Parse the command
    parsed = language_processor.parse_robot_command(cmd)
    print(f"Parsed: {parsed['parsed_command']}")
    print(f"Confidence: {parsed['confidence']:.2f}")
    
    # Execute the command
    if parsed['confidence'] > 0.3:
        result = command_executor.execute_command(parsed)
        print(f"Execution result: {result['message']}")
        if result['success']:
            state = result['robot_state']
            print(f"  Robot position: {state['position'][:2]}")
            print(f"  Holding object: {state['held_object'] if state['is_holding_object'] else 'None'}")
    else:
        print("Command confidence too low, skipping execution")
```

## Dialog Management and Context Understanding

### Maintaining Conversation Context

Advanced conversational systems need to maintain context across turns to provide coherent responses:

```python
import json
import time
from typing import Dict, List, Any, Optional
from collections import deque

class ContextManager:
    """Manages conversation context and coherence"""
    def __init__(self, max_context_length: int = 10):
        self.max_context_length = max_context_length
        self.context_history = deque(maxlen=max_context_length)
        self.current_topic = "greeting"
        self.topic_stack = []
        self.user_model = {
            "preferences": {},
            "conversation_style": "neutral",
            "topic_interests": set(),
            "personality_traits": {
                "patience": 0.7,
                "helpfulness": 0.9,
                "formality": 0.6
            }
        }
    
    def update_context(self, user_input: str, parsed_intent: str, entities: Dict[str, Any], 
                      robot_response: str) -> None:
        """Update the conversation context with new information"""
        turn_context = {
            "timestamp": time.time(),
            "user_input": user_input,
            "parsed_intent": parsed_intent,
            "entities": entities,
            "robot_response": robot_response,
            "topic": self.current_topic
        }
        
        self.context_history.append(turn_context)
        
        # Update topic if it changed
        new_topic = self._infer_topic(user_input, parsed_intent, entities)
        if new_topic != self.current_topic:
            self.topic_stack.append(self.current_topic)
            self.current_topic = new_topic
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of the current context"""
        if not self.context_history:
            return {
                "current_topic": self.current_topic,
                "conversation_length": 0,
                "active_entities": {},
                "user_preferences": self.user_model["preferences"]
            }
        
        # Identify recently mentioned entities
        active_entities = {}
        for turn in list(self.context_history)[-3:]:  # Look at last 3 turns
            for entity_key, entity_val in turn["entities"].items():
                if entity_key not in active_entities:
                    active_entities[entity_key] = []
                active_entities[entity_key].append(entity_val)
        
        return {
            "current_topic": self.current_topic,
            "conversation_length": len(self.context_history),
            "active_entities": active_entities,
            "user_preferences": self.user_model["preferences"],
            "topic_history": self.topic_stack[-5:]  # Last 5 topics
        }
    
    def resolve_coreferences(self, text: str) -> str:
        """Resolve coreferences like 'it', 'that', 'there' based on context"""
        if not self.context_history:
            return text
        
        # Look for pronouns and demonstratives to resolve
        resolved_text = text.lower()
        
        # Resolve "it" references
        if " it " in resolved_text or resolved_text.startswith("it "):
            # Find the last mentioned object
            for i in range(len(self.context_history)-1, -1, -1):
                entities = self.context_history[i]["entities"]
                if "object" in entities:
                    obj = entities["object"]
                    resolved_text = resolved_text.replace(" it ", f" {obj} ")
                    resolved_text = resolved_text.replace("it ", f"{obj} ")
                    break
        
        # Resolve "that" references
        if " that " in resolved_text:
            # Find the last mentioned object or location
            for i in range(len(self.context_history)-1, -1, -1):
                entities = self.context_history[i]["entities"]
                if "object" in entities:
                    obj = entities["object"]
                    resolved_text = resolved_text.replace(" that ", f" {obj} ")
                    break
                elif "location" in entities:
                    loc = entities["location"]
                    resolved_text = resolved_text.replace(" that ", f" {loc} ")
                    break
        
        # Resolve "there" references
        if " there " in resolved_text:
            # Find the last mentioned location
            for i in range(len(self.context_history)-1, -1, -1):
                entities = self.context_history[i]["entities"]
                if "location" in entities:
                    loc = entities["location"]
                    resolved_text = resolved_text.replace(" there ", f" {loc} ")
                    break
        
        return resolved_text
    
    def _infer_topic(self, user_input: str, parsed_intent: str, entities: Dict[str, Any]) -> str:
        """Infer the current conversation topic"""
        topic_keywords = {
            "navigation": ["go", "move", "walk", "navigate", "location", "direction"],
            "manipulation": ["pick", "grab", "place", "hold", "object", "item"],
            "information": ["what", "where", "who", "when", "how", "question"],
            "social": ["hello", "hi", "good", "friend", "chat", "talk", "story"],
            "greeting": ["hello", "hi", "hey", "greeting"],
            "farewell": ["goodbye", "bye", "see", "farewell"]
        }
        
        text_lower = user_input.lower()
        
        # Check intent-based topics first
        intent_topic_map = {
            "greeting": "greeting",
            "farewell": "farewell", 
            "navigation": "navigation",
            "manipulation": "manipulation",
            "information_request": "information",
            "social_interaction": "social"
        }
        
        if parsed_intent in intent_topic_map:
            return intent_topic_map[parsed_intent]
        
        # If no intent match, use keyword-based inference
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return topic
        
        # Default to information if no clear topic
        return "information"
    
    def update_user_model(self, user_input: str, robot_response: str) -> None:
        """Update the user model based on conversation"""
        text_lower = user_input.lower()
        
        # Detect politeness
        if any(word in text_lower for word in ["please", "thank you", "thanks", "excuse me"]):
            self.user_model["personality_traits"]["formality"] = min(
                1.0, self.user_model["personality_traits"]["formality"] + 0.1
            )
        
        # Detect urgency
        if any(word in text_lower for word in ["urgent", "quickly", "hurry", "fast", "now"]):
            self.user_model["personality_traits"]["patience"] = max(
                0.0, self.user_model["personality_traits"]["patience"] - 0.1
            )
        
        # Detect helpfulness indicators
        if any(word in text_lower for word in ["help", "assist", "support"]):
            self.user_model["personality_traits"]["helpfulness"] = min(
                1.0, self.user_model["personality_traits"]["helpfulness"] + 0.05
            )

class AdvancedDialogManager:
    """Advanced dialog management with context awareness"""
    def __init__(self):
        self.context_manager = ContextManager()
        
        # Response templates that can use context
        self.response_templates = {
            "greeting": [
                "Hello! How can I assist you today?",
                "Hi there! What would you like to do?",
                "Good day! How may I help you?"
            ],
            "navigation": [
                "I can help you navigate to {location}. Is that correct?",
                "I'm ready to move to {location}. Should I proceed?",
                "I can go to {location}. Do you need anything else?"
            ],
            "manipulation": [
                "I can pick up the {object}. Where should I place it?",
                "I see the {object}. Would you like me to grab it?",
                "I'm ready to handle the {object}. What would you like me to do?"
            ],
            "information_request": [
                "I can provide information about {topic}. What specifically would you like to know?",
                "Regarding {topic}, I know quite a bit. What aspect interests you?",
                "I'm knowledgeable about {topic}. How can I help?"
            ],
            "social_interaction": [
                "I enjoy chatting about {topic}. What's your perspective?",
                "Let's discuss {topic}. What are your thoughts?",
                "I'd be happy to talk about {topic}. Tell me more."
            ],
            "follow_up": [
                "Did you need help with anything else?",
                "Is there anything else I can assist you with?",
                "Would you like to continue our conversation?"
            ]
        }
        
        # Default responses when context is insufficient
        self.default_responses = [
            "I'm here to help. Could you please specify what you'd like me to do?",
            "I can assist with navigation, object manipulation, and conversation. What do you need?",
            "How can I be of service today?"
        ]
    
    def generate_contextual_response(self, user_input: str, parsed_intent: str, 
                                   entities: Dict[str, Any]) -> str:
        """Generate a response based on current context"""
        # Resolve coreferences first
        resolved_input = self.context_manager.resolve_coreferences(user_input)
        
        # Update user model
        self.context_manager.update_user_model(user_input, "")
        
        # Get current context
        context = self.context_manager.get_context_summary()
        
        # If we have entities, try to use them in the response
        if entities and parsed_intent in self.response_templates:
            response_template = self.response_templates[parsed_intent][0]  # Use first template
            
            # Try to fill in any placeholders
            filled_response = response_template.format(**entities)
            
            # If no entities but we have context, use context info
            if filled_response == response_template:
                if context["active_entities"]:
                    # Use the most recently mentioned entity
                    latest_entities = list(context["active_entities"].values())[-1]
                    if latest_entities:
                        entity_val = latest_entities[-1]  # Most recent value
                        filled_response = response_template.format(
                            object=entity_val, 
                            location=entity_val,
                            topic=entity_val
                        )
        
        elif parsed_intent in self.response_templates:
            # Use template without filling
            filled_response = self.response_templates[parsed_intent][0]
        else:
            # Use default response
            import random
            filled_response = random.choice(self.default_responses)
        
        # Add follow-up if this is a natural place for one
        if self._should_add_followup(parsed_intent, context):
            followup = self.response_templates["follow_up"][0]
            filled_response += f" {followup}"
        
        # Update context with the response
        self.context_manager.update_context(user_input, parsed_intent, entities, filled_response)
        
        return filled_response
    
    def _should_add_followup(self, intent: str, context: Dict[str, Any]) -> bool:
        """Determine if a follow-up question should be added"""
        # Add follow-up after completing a task
        followup_triggers = ["navigation", "manipulation"]
        
        if intent in followup_triggers and context["conversation_length"] > 1:
            return True
        
        # Add follow-up after providing information
        if intent == "information_request":
            return True
        
        return False

# Example usage
advanced_dialog = AdvancedDialogManager()

print("Testing advanced contextual responses:")

# Simulate a conversation
conversation_turns = [
    ("Hello there!", "greeting", {}),
    ("Can you go to the kitchen?", "navigation", {"location": "kitchen"}),
    ("Yes, that's right", "confirmation", {}),
    ("I want to pick up the red cup", "manipulation", {"object": "red cup"}),
    ("Where should I put it?", "information_request", {"object": "red cup"}),
    ("Place it on the table", "manipulation", {"location": "table"}),
    ("Thank you for your help", "gratitude", {})
]

for i, (user_input, intent, entities) in enumerate(conversation_turns):
    print(f"\nTurn {i+1}:")
    print(f"User: '{user_input}'")
    print(f"Intent: {intent}")
    print(f"Entities: {entities}")
    
    response = advanced_dialog.generate_contextual_response(user_input, intent, entities)
    print(f"Robot: '{response}'")
    
    context = advanced_dialog.context_manager.get_context_summary()
    print(f"Context: Topic='{context['current_topic']}', Length={context['conversation_length']}")
```

## Social Interaction and Multimodal Communication

### Integrating Non-Verbal Communication

Effective humanoid conversational systems must integrate verbal and non-verbal communication:

```python
import numpy as np
import math
from typing import Dict, List, Any, Tuple
from enum import Enum

class GazeBehavior(Enum):
    MAINTAIN = "maintain"
    SHIFT = "shift"
    AVOID = "avoid"
    FOLLOW = "follow"

class GestureType(Enum):
    WAVE = "wave"
    POINT = "point"
    NOD = "nod"
    SHAKE_HEAD = "shake_head"
    THUMBS_UP = "thumbs_up"
    GREETING = "greeting"
    EMPHASIS = "emphasis"

class EmotionalState(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    SURPRISED = "surprised"
    ANGRY = "angry"
    CONFUSED = "confused"

class SocialInteractionManager:
    """Manages non-verbal communication and social behaviors"""
    def __init__(self):
        self.current_gaze_target = None
        self.current_emotion = EmotionalState.NEUTRAL
        self.face_tracking_enabled = True
        self.gesture_queue = []
        self.social_rules = {
            "personal_space": 1.0,  # meters
            "gaze_duration_min": 0.5,  # seconds
            "gaze_duration_max": 3.0,  # seconds
            "conversational_distance": 1.2  # meters for conversation
        }
    
    def determine_gaze_target(self, context: Dict[str, Any], user_position: np.ndarray) -> np.ndarray:
        """Determine where the robot should look based on context"""
        # Priority order for gaze targets:
        # 1. Speaking person's face (if face detected)
        # 2. Person's general direction
        # 3. Object being discussed
        # 4. Direction of movement
        
        if context.get("speaker_position") is not None:
            return np.array(context["speaker_position"])
        
        if user_position is not None:
            return user_position
        
        if context.get("discussed_object_position") is not None:
            return np.array(context["discussed_object_position"])
        
        # Default to forward direction if no specific target
        return np.array([1.0, 0.0, 0.0])  # Looking forward
    
    def generate_greeting_behavior(self, user_distance: float) -> List[Dict[str, Any]]:
        """Generate appropriate greeting behavior based on distance"""
        behaviors = []
        
        if user_distance > self.social_rules["conversational_distance"]:
            # Wave to get attention
            behaviors.append({
                "type": "gesture",
                "gesture": GestureType.WAVE,
                "duration": 2.0
            })
            behaviors.append({
                "type": "gaze",
                "behavior": GazeBehavior.FOLLOW,
                "target": "user"
            })
        
        elif user_distance < self.social_rules["personal_space"]:
            # Give space and nod
            behaviors.append({
                "type": "gaze",
                "behavior": GazeBehavior.MAINTAIN,
                "target": "user"
            })
            behaviors.append({
                "type": "gesture", 
                "gesture": GestureType.NOD,
                "duration": 1.0
            })
        else:
            # Normal greeting at conversational distance
            behaviors.extend([
                {
                    "type": "gaze",
                    "behavior": GazeBehavior.MAINTAIN,
                    "target": "user"
                },
                {
                    "type": "gesture",
                    "gesture": GestureType.GREETING,
                    "duration": 1.5
                }
            ])
        
        return behaviors
    
    def generate_response_gestures(self, intent: str) -> List[Dict[str, Any]]:
        """Generate appropriate gestures for different intents"""
        gesture_map = {
            "greeting": [GestureType.NOD, GestureType.WAVE],
            "navigation": [GestureType.POINT],
            "manipulation": [GestureType.EMPHASIS],
            "information_request": [GestureType.NOD],
            "social_interaction": [GestureType.EMPHASIS],
            "farewell": [GestureType.WAVE, GestureType.NOD]
        }
        
        if intent in gesture_map:
            gestures = gesture_map[intent]
            return [{"type": "gesture", "gesture": g, "duration": 1.0} for g in gestures]
        
        return []
    
    def adjust_emotional_expression(self, context: Dict[str, Any]) -> EmotionalState:
        """Adjust robot's emotional expression based on conversation context"""
        user_emotion = context.get("user_emotion", "neutral")
        
        if user_emotion in ["happy", "excited"]:
            return EmotionalState.HAPPY
        elif user_emotion in ["sad", "upset"]:
            return EmotionalState.SAD
        elif user_emotion in ["surprised", "amazed"]:
            return EmotionalState.SURPRISED
        elif user_emotion in ["confused", "uncertain"]:
            return EmotionalState.CONFUSED
        
        # If user emotion is neutral or unknown, maintain neutral
        return EmotionalState.NEUTRAL
    
    def coordinate_multimodal_response(self, text_response: str, intent: str, 
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate speech, gestures, and gaze for multimodal response"""
        response = {
            "speech": text_response,
            "gaze_target": None,
            "gestures": [],
            "emotional_state": self.current_emotion
        }
        
        # Determine gaze target
        user_pos = context.get("user_position", [0, 1, 0])  # Default to user in front
        response["gaze_target"] = self.determine_gaze_target(context, user_pos)
        
        # Generate appropriate gestures
        response["gestures"] = self.generate_response_gestures(intent)
        
        # Adjust emotional expression
        response["emotional_state"] = self.adjust_emotional_expression(context)
        
        return response

class ConversationalHumanoid:
    """Complete conversational humanoid system with multimodal capabilities"""
    def __init__(self):
        self.language_processor = RobotLanguageProcessor()
        self.command_executor = RobotCommandExecutor()
        self.dialog_manager = AdvancedDialogManager()
        self.social_manager = SocialInteractionManager()
        
        # Robot state
        self.robot_position = np.array([0.0, 0.0, 0.0])
        self.is_active = True
        self.current_user_position = np.array([1.0, 0.0, 0.0])
        
        # Interaction history
        self.interaction_log = []
    
    def process_conversation_turn(self, user_input: str, user_position: np.ndarray = None) -> Dict[str, Any]:
        """Process a complete turn of conversation with multimodal response"""
        if user_position is not None:
            self.current_user_position = user_position
        
        # Parse the user input
        parsed_result = self.language_processor.parse_robot_command(user_input)
        
        # Generate dialog response
        dialog_response = self.dialog_manager.generate_contextual_response(
            user_input, 
            parsed_result["action"], 
            parsed_result["parsed_command"]
        )
        
        # Prepare context for social manager
        context = {
            "user_position": self.current_user_position.tolist(),
            "conversation_topic": self.dialog_manager.context_manager.current_topic,
            "user_emotion": "neutral",  # Would come from emotion detection
            "last_robot_response": dialog_response
        }
        
        # Generate multimodal response
        multimodal_response = self.social_manager.coordinate_multimodal_response(
            dialog_response,
            parsed_result["action"],
            context
        )
        
        # Execute robot command if applicable
        execution_result = self.command_executor.execute_command(parsed_result)
        
        # Log the interaction
        interaction = {
            "timestamp": time.time(),
            "user_input": user_input,
            "parsed_command": parsed_result,
            "dialog_response": dialog_response,
            "multimodal_response": multimodal_response,
            "execution_result": execution_result,
            "user_position": self.current_user_position.tolist()
        }
        self.interaction_log.append(interaction)
        
        return {
            "speech_response": dialog_response,
            "multimodal_response": multimodal_response,
            "execution_result": execution_result,
            "parsed_command": parsed_result
        }
    
    def handle_continuous_interaction(self, max_turns: int = 10) -> List[Dict[str, Any]]:
        """Simulate a continuous interaction over multiple turns"""
        print("Starting continuous interaction simulation...")
        interactions = []
        
        # Simulated user inputs for the demo
        simulated_inputs = [
            "Hello robot!",
            "Please come here.",
            "Can you navigate to the kitchen?",
            "I need you to pick up the red cup.",
            "Where is the cup now?",
            "Put the cup on the table.",
            "How are you doing?",
            "Tell me about your day.",
            "Can you help me with something?",
            "Thank you, goodbye!"
        ]
        
        for i in range(min(max_turns, len(simulated_inputs))):
            user_input = simulated_inputs[i]
            print(f"\nTurn {i+1}: User says '{user_input}'")
            
            result = self.process_conversation_turn(
                user_input, 
                np.array([1.0, 0.0, 0.0])  # Fixed user position for simulation
            )
            
            print(f"Robot says: '{result['speech_response']}'")
            print(f"Gestures: {[g['gesture'] for g in result['multimodal_response']['gestures']]}")
            
            if result['execution_result']['message']:
                print(f"Execution: {result['execution_result']['message']}")
            
            interactions.append(result)
        
        return interactions

# Example usage
humanoid = ConversationalHumanoid()

# Run a continuous interaction simulation
interactions = humanoid.handle_continuous_interaction(max_turns=5)

print(f"\nCompleted {len(interactions)} interaction turns")
print(f"Last robot position: {humanoid.command_executor.robot_position[:2]}")
print(f"Holding object: {humanoid.command_executor.held_object}")
```

## Exercises

1. **Conversational System Design**: Design a complete conversational system for a humanoid robot that includes speech recognition, natural language understanding, dialog management, and response generation. Implement the system and evaluate its performance.

2. **Context-Aware Dialog**: Create a dialog system that maintains conversation context over multiple turns and uses this context to provide coherent and relevant responses.

3. **Multimodal Interaction**: Implement a system that coordinates speech with gestures, gaze, and other non-verbal behaviors for more natural human-robot interaction.

4. **Intent Classification**: Develop an intent classification system that can accurately identify user intentions from natural language input with a focus on robotics commands.

5. **Social Behavior Integration**: Create a system that generates appropriate social behaviors (gaze, gestures, expressions) based on the conversation content and context.

6. **Error Handling**: Implement robust error handling for cases where the robot doesn't understand user input or cannot execute commands.

7. **Personalization**: Create a conversational system that learns and adapts to individual users' preferences and interaction styles over time.

## Summary

Conversational systems for humanoid robots involve complex integration of speech recognition, natural language understanding, dialog management, and multimodal communication. These systems must handle the nuances of natural human language while coordinating with the robot's physical capabilities and social behaviors. Success requires balancing natural, human-like interaction with the robot's computational limitations and safety requirements. Advanced systems incorporate context awareness, emotional intelligence, and personalization to create more engaging and effective human-robot interactions.