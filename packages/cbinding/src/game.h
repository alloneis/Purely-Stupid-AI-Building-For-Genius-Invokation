#pragma once
#ifndef GITCG_GAME_H
#define GITCG_GAME_H

#include <optional>
#include <v8.h>

#include "gitcg/gitcg.h"
#include "state.h"
#include "object.h"

namespace gitcg {
inline namespace v1_0 {

class Game final : public Object {
  int game_id;

  void* player_data[2]{};
  gitcg_rpc_handler rpc_handler[2]{};
  gitcg_notification_handler notification_handler[2]{};
  gitcg_io_error_handler io_error_handler[2]{};

public:
  Game(Environment* environment, int game_id, v8::Local<v8::Object> instance);

  void* get_player_data(int who) const noexcept {
    return player_data[who];
  }
  void set_player_data(int who, void* data) noexcept {
    player_data[who] = data;
  }
  gitcg_rpc_handler get_rpc_handler(int who) const noexcept {
    return rpc_handler[who];
  }
  void set_rpc_handler(int who, gitcg_rpc_handler handler) noexcept {
    rpc_handler[who] = handler;
  }
  gitcg_notification_handler get_notification_handler(int who) const noexcept {
    return notification_handler[who];
  }
  void set_notification_handler(int who,
                                gitcg_notification_handler handler) noexcept {
    notification_handler[who] = handler;
  }
  gitcg_io_error_handler get_io_error_handler(int who) const noexcept {
    return io_error_handler[who];
  }
  void set_io_error_handler(int who, gitcg_io_error_handler handler) noexcept {
    io_error_handler[who] = handler;
  }

  int get_status() const;
  char* get_error() const;
  bool is_resumable() const;
  State& get_state();
  int get_winner() const;

  void set_attribute(int key, int value);
  int get_attribute(int key) const;

  void step();
  void giveup(int who);
};

}  // namespace v1_0
}  // namespace gitcg

#endif
