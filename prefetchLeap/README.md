# Description
This is an apache traffic server plugin.
Recomended remap config:
```
map http://www.example.com/ http://server1.example.com/ \
@plugin=/opt/ts/libexec/trafficserver/cachekey.so @pparam=--remove-all-params=true \
@plugin=/opt/ts/libexec/trafficserver/myprefetch.so \
@pparam=--front=true \
@pparam=--fetch-policy=simple \
@pparam=--fetch-path-pattern=/(.*-)(\d+)(.*)/$1{$2+1}$3/ \
@pparam=--fetch-count=1 \
@pparam=--exact-match=true \
@pparam=--log-name=prefetch
```

