import 'dart:math';

const List<String> messages = [
  "Don't be surprised when a crack in the ice appears under your feet.",
  "'OOF'[someone being hit]",
  "Hey! Teachers! Leave them kids alone!",
  "If you don't eat yer meat, you can't have any pudding. How can you have any pudding if you don't eat yer meat?",
  "'Yes, a collect call for Mrs. Floyd from Mr. Floyd.\n Will you accept the charges from United States?'",
  "I can feel one of my turns coming on.",
  "Would you like to call the cops?\n Do you think it's time I stopped?",
  "Why are you running away?",
  "And I dont need no drugs to calm me.",
  "All in all it was all just bricks in the wall.",
  "He could not break free.\n And the worms ate into his brain."
  "Got those swollen hand blues.",
  "There is no pain you are receding",
  "Are there any queers in the theater tonight?",
  "If I had my way, I'd have all of you shot!",
  "'Go on Judge! Shit on him!'",
  "Tear down the wall!",
  "Some stagger and fall, after all it's not easy\n Banging your heart against some mad bugger's wall.",
];

final Random _random = Random();

String getRandomMessage() {
  return messages[_random.nextInt(messages.length)];
}
