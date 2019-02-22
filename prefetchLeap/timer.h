//
// Created by wangc on 2018/12/24.
//

#ifndef MYPREFETCH_TIMER_H
#define MYPREFETCH_TIMER_H

#endif //MYPREFETCH_TIMER_H
//
// Created by wangc on 2018/12/24.
//

//timer
#include "tscpp/api/AsyncTimer.h"
using namespace atscppapi;


class TimerEventReceiver : public AsyncReceiver<AsyncTimer>
{
public:
    TimerEventReceiver(AsyncTimer::Type type, int period_in_ms, BgFetchState *_state, int initial_period_in_ms = 0, int max_instances = 0,
                       bool cancel = false)
            : max_instances_(max_instances), instance_count_(0), type_(type), cancel_(cancel)
    {
        _period_in_ms = period_in_ms;
        timer_ = new AsyncTimer(type, period_in_ms, initial_period_in_ms);
        Async::execute<AsyncTimer>(this, timer_, std::shared_ptr<Mutex>()); // letting the system create the mutex
        this->_state = _state;
    }

    void
    handleAsyncComplete(AsyncTimer &timer) override
    {
        if(_period_in_ms == 1000){ //per second
            _state->reset();
        } else if(_period_in_ms == 3600000){ //per hour
            _state->ResetQoeGain();
        }

        if ((type_ == AsyncTimer::TYPE_ONE_OFF) || (max_instances_ && (++instance_count_ == max_instances_))) {
            cancel_ ? timer_->cancel() : delete this;
        }
    }

    ~TimerEventReceiver() override { delete timer_; }

private:
    int max_instances_;
    int instance_count_;
    AsyncTimer::Type type_;
    AsyncTimer *timer_;
    bool cancel_;
    BgFetchState *_state;
    int _period_in_ms;
};
