#include "aurora_ui/aurora_ui.hpp"
#include "aurora_ui/components/aurora_text.hpp"
#include "aurora_ui/components/aurora_circle.hpp"
#include "aurora_ui/components/aurora_panel.hpp"
#include "aurora_ui/components/aurora_segment.hpp"
#include "aurora_ui/utils/aurora_theme_settings.hpp"
#include "aurora_debug/aurora_debug_json.hpp"
#include "aurora_debug/aurora_debug_server.hpp"

#include "aurora_engine/utils/log.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

// Scenes streamed by OrbitML (see stream/scenes.py), as Custom (3) messages with flat JSON payloads:
//   {"clear": 1}                                   removes every object, the titles and the panel entries
//   {"layout": "both" | "space" | "map"}           space view on top of the map, or one of them alone
//   {"titles": 1, "space": "<text>", "map": "<text>"}
//   {"camera": [right, up, back], "center": [x, y, z], "extent": km}
//        rows of the space to view rotation (back points toward the viewer), the point at the center
//        of the space view, and the distance from that center to the edge of the view
//   {"earth": [m00, m01, ..., m22], "surface": 0 | 1}
//        the Earth fixed to space rotation row by row, and whether to draw the grid and the coastlines
//        on the globe. Without an "earth" message the space view has no Earth
//   {"coastline": [latitude, longitude, ...]}      one per coastline, in degrees
//   {"id": "<name>", "line": [x, y, z, ...], "color": "<name>", "size": pixels}    polyline in space, km
//   {"id": "<name>", "point": [x, y, z], "color": "<name>", "size": pixels, "label": "<text>"}
//   {"id": "<name>", "dots": [x, y, z, ...], "color": "<name>", "size": pixels}    points in space, km
//   {"id": "<name>", "track": [latitude, longitude, ...], "color": "<name>", "size": pixels}    map polyline
//   {"id": "<name>", "spots": [latitude, longitude, ...], "colors": [r, g, b, ...], "size": pixels}
//   {"remove": "<id>"}
// Sending an id again replaces the object. Watch (1) messages fill the panel.
// Key presses are sent back as Command (4) messages, {"command": "<name>"}:
//   SPACE -> pause, UP -> faster, DOWN -> slower, R -> restart, N -> next, P -> previous

class OrbitViewer : public aurora::AuroraUI {
    public:
        OrbitViewer() : AuroraUI{"OrbitML"} {}

    protected:
        void onSetup() override {
            const auto& theme = aurora::AuroraThemeSettings::get();

            earth = create<aurora::AuroraCircle>(1.f, glm::mix(theme.BACKGROUND, theme.BLUE, EARTH_TINT));
            earth->setLayer(EARTH_LAYER);
            earth->setHidden(true);

            space_title = create<aurora::AuroraText>(" ", TITLE_SIZE, theme.TEXT_SECONDARY);
            map_title = create<aurora::AuroraText>(" ", TITLE_SIZE, theme.TEXT_SECONDARY);

            // One line every GRID_STEP degrees, borders included
            for (int i = 0; i <= 360 / GRID_STEP; ++i) meridians.push_back(mapGridLine());
            for (int i = 0; i <= 180 / GRID_STEP; ++i) parallels.push_back(mapGridLine());

            // The same lines on the globe, sampled every GLOBE_GRID_SAMPLE degrees. Parallels skip the poles
            for (int longitude = -180; longitude < 180; longitude += GRID_STEP) {
                GlobeLine line;
                for (int latitude = -90; latitude <= 90; latitude += GLOBE_GRID_SAMPLE) line.units.push_back(unitVector(latitude, longitude));
                addGlobeLine(globe_grid, std::move(line), theme.GRAY);
            }
            for (int latitude = -90 + GRID_STEP; latitude < 90; latitude += GRID_STEP) {
                GlobeLine line;
                for (int longitude = -180; longitude <= 180; longitude += GLOBE_GRID_SAMPLE) line.units.push_back(unitVector(latitude, longitude));
                addGlobeLine(globe_grid, std::move(line), theme.GRAY);
            }

            newPanel();

            server.onDisconnect([this](const std::string&) {
                if (server.getClientCount() == 0) clear();
            });

            server.onMessage([this](aurora::debug::AuroraDebugSession&, const aurora::debug::DebugMessage& msg) {
                aurora::debug::WatchMessage watch;
                if (msg.type == aurora::debug::MessageType::Custom) receive(msg.payload);
                else if (msg.type == aurora::debug::MessageType::Watch && aurora::debug::WatchMessage::parse(msg.payload, watch)) {
                    panel->getSection(watch.section).setEntry(watch.name, watch.value);
                }
            });

            server.start();
        }

        void onUpdate(float /*dt*/) override {
            server.poll();
            layout();
        }

        void onKey(int key, int /*mods*/) override {
            switch (key) {
                case GLFW_KEY_SPACE: sendCommand("pause"); break;
                case GLFW_KEY_UP: sendCommand("faster"); break;
                case GLFW_KEY_DOWN: sendCommand("slower"); break;
                case GLFW_KEY_R: sendCommand("restart"); break;
                case GLFW_KEY_N: sendCommand("next"); break;
                case GLFW_KEY_P: sendCommand("previous"); break;
                default: break;
            }
        }

    private:
        static constexpr uint16_t PORT = 9000;
        static constexpr float PANEL_WIDTH = 420.f;
        static constexpr float MARGIN = 50.f;
        static constexpr float TITLE_SIZE = 16.f;
        static constexpr float LABEL_SIZE = 20.f;
        static constexpr int GRID_STEP = 30;  // degrees between grid lines
        static constexpr int GLOBE_GRID_SAMPLE = 5;  // degrees between points of the globe grid lines
        static constexpr float GRID_THICKNESS = 1.f;
        static constexpr float COASTLINE_THICKNESS = 1.f;
        static constexpr float EARTH_TINT = 0.12f;
        static constexpr float EARTH_RADIUS = 6378.137f;  // km
        static constexpr float JUMP = 180.f;  // degrees of longitude: a longer step on the map crosses the antimeridian

        // Drawn back to front. What is behind the Earth goes under it, the opaque disk hides it
        static constexpr int BACKGROUND_LAYER = 0;
        static constexpr int BEHIND_EARTH_LAYER = 1;
        static constexpr int EARTH_LAYER = 2;
        static constexpr int SURFACE_LAYER = 3;
        static constexpr int OBJECT_LAYER = 4;
        static constexpr int POINT_LAYER = 5;
        static constexpr int PANEL_LAYER = 6;

        struct View {
            glm::vec2 origin;  // top left
            glm::vec2 size;
            bool shown = false;
        };

        // A line on the surface of the globe
        struct GlobeLine {
            std::vector<glm::vec3> units;  // Earth fixed unit vectors
            std::vector<std::shared_ptr<aurora::AuroraSegment>> segments;  // one between each pair of points
        };

        struct Coastline {
            std::vector<glm::vec2> points;  // (longitude, latitude) in degrees
            std::vector<std::shared_ptr<aurora::AuroraSegment>> segments;  // on the map
        };

        enum class Kind { Line, Point, Dots, Track, Spots };

        struct Object {
            Kind kind;
            std::vector<glm::vec3> points;  // km in space, or (longitude, latitude, 0) on the map
            std::string color;
            std::vector<float> colors;  // spots only, r g b per spot
            float size;
            std::string label;
            std::vector<std::shared_ptr<aurora::AuroraSegment>> segments;
            std::vector<std::shared_ptr<aurora::AuroraCircle>> circles;
            std::shared_ptr<aurora::AuroraText> text;
        };

        static glm::vec3 unitVector(float latitude, float longitude) {
            latitude = glm::radians(latitude);
            longitude = glm::radians(longitude);
            return {std::cos(latitude) * std::cos(longitude), std::cos(latitude) * std::sin(longitude), std::sin(latitude)};
        }

        static glm::vec4 namedColor(const std::string& name) {
            const auto& theme = aurora::AuroraThemeSettings::get();
            if (name == "orange") return theme.ORANGE;
            if (name == "purple") return theme.PURPLE;
            if (name == "green") return theme.SUCCESS;
            if (name == "red") return theme.ERROR;
            if (name == "gray") return theme.DELIMITER;
            if (name == "white") return theme.TEXT_PRIMARY;
            return theme.BLUE;
        }

        static std::vector<glm::vec3> triplets(const std::vector<float>& values) {
            std::vector<glm::vec3> points;
            for (size_t i = 0; i + 2 < values.size(); i += 3) points.push_back({values[i], values[i + 1], values[i + 2]});
            return points;
        }

        // (latitude, longitude) pairs to (longitude, latitude, 0)
        static std::vector<glm::vec3> pairs(const std::vector<float>& values) {
            std::vector<glm::vec3> points;
            for (size_t i = 0; i + 1 < values.size(); i += 2) points.push_back({values[i + 1], values[i], 0.f});
            return points;
        }

        void newPanel() {
            if (panel) destroy(panel);
            panel = create<aurora::AuroraPanel>(PANEL_WIDTH);
            panel->setLayer(PANEL_LAYER);
        }

        std::shared_ptr<aurora::AuroraSegment> mapGridLine() {
            auto line = create<aurora::AuroraSegment>(aurora::AuroraThemeSettings::get().GRAY);
            line->setLayer(BACKGROUND_LAYER);
            return line;
        }

        void addGlobeLine(std::vector<GlobeLine>& lines, GlobeLine line, glm::vec4 color) {
            for (size_t i = 1; i < line.units.size(); ++i) {
                auto segment = create<aurora::AuroraSegment>(color);
                segment->setLayer(SURFACE_LAYER);
                line.segments.push_back(segment);
            }
            lines.push_back(std::move(line));
        }

        void receive(const std::string& payload) {
            using namespace aurora::debug::json;

            if (!getString(payload, "clear").empty()) return clear();

            std::string id = getString(payload, "id");
            if (!id.empty()) return receiveObject(id, payload);

            std::string removed = getString(payload, "remove");
            if (!removed.empty()) {
                auto found = objects.find(removed);
                if (found != objects.end()) {
                    destroyObject(found->second);
                    objects.erase(found);
                }
                return;
            }

            std::string mode = getString(payload, "layout");
            if (!mode.empty()) {
                layout_mode = mode;
                window_size = {0.f, 0.f};  // forces layout() to place everything again
                return;
            }

            if (!getString(payload, "titles").empty()) {
                space_title->setText(getString(payload, "space") + " ");
                map_title->setText(getString(payload, "map") + " ");
                return;
            }

            std::vector<float> coastline = getNumberArray(payload, "coastline");
            if (!coastline.empty()) return addCoastline(coastline);

            std::vector<float> rotation = getNumberArray(payload, "earth");
            if (rotation.size() == 9) {
                for (int row = 0; row < 3; ++row) {
                    for (int column = 0; column < 3; ++column) earth_to_space[column][row] = rotation[3 * row + column];  // glm is column major
                }
                has_earth = true;
                surface = getNumber(payload, "surface", 1.0) != 0.0;
                placeSpace();
                return;
            }

            std::vector<float> camera_rows = getNumberArray(payload, "camera");
            if (camera_rows.size() == 9) {
                for (int row = 0; row < 3; ++row) {
                    for (int column = 0; column < 3; ++column) camera[column][row] = camera_rows[3 * row + column];
                }
                std::vector<float> center = getNumberArray(payload, "center");
                if (center.size() == 3) camera_center = {center[0], center[1], center[2]};
                extent = static_cast<float>(getNumber(payload, "extent", extent));
                placeSpace();
            }
        }

        void receiveObject(const std::string& id, const std::string& payload) {
            using namespace aurora::debug::json;

            Object object;
            object.color = getString(payload, "color");
            object.size = static_cast<float>(getNumber(payload, "size", 2.0));
            object.label = getString(payload, "label");
            if (std::vector<float> values = getNumberArray(payload, "line"); !values.empty()) {
                object.kind = Kind::Line;
                object.points = triplets(values);
            } else if (values = getNumberArray(payload, "point"); values.size() == 3) {
                object.kind = Kind::Point;
                object.points = triplets(values);
            } else if (values = getNumberArray(payload, "dots"); !values.empty()) {
                object.kind = Kind::Dots;
                object.points = triplets(values);
            } else if (values = getNumberArray(payload, "track"); !values.empty()) {
                object.kind = Kind::Track;
                object.points = pairs(values);
            } else if (values = getNumberArray(payload, "spots"); !values.empty()) {
                object.kind = Kind::Spots;
                object.points = pairs(values);
                object.colors = getNumberArray(payload, "colors");
            } else {
                return;
            }

            // Same shape and look: only the points move, the components are kept
            auto found = objects.find(id);
            if (found != objects.end()) {
                Object& old = found->second;
                if (old.kind == object.kind && old.points.size() == object.points.size() && old.color == object.color
                    && old.size == object.size && old.label == object.label && object.kind != Kind::Spots) {
                    old.points = std::move(object.points);
                    placeObject(old);
                    return;
                }
                destroyObject(old);
                objects.erase(found);
            }

            const auto& theme = aurora::AuroraThemeSettings::get();
            glm::vec4 color = namedColor(object.color);
            switch (object.kind) {
                case Kind::Line:
                case Kind::Track:
                    for (size_t i = 1; i < object.points.size(); ++i) {
                        object.segments.push_back(create<aurora::AuroraSegment>(color));
                        object.segments.back()->setLayer(OBJECT_LAYER);
                    }
                    break;
                case Kind::Point:
                case Kind::Dots:
                    for (size_t i = 0; i < object.points.size(); ++i) {
                        object.circles.push_back(create<aurora::AuroraCircle>(object.size, color));
                        object.circles.back()->setLayer(POINT_LAYER);
                    }
                    break;
                case Kind::Spots:
                    for (size_t i = 0; i < object.points.size(); ++i) {
                        glm::vec4 spot = 3 * i + 2 < object.colors.size()
                            ? glm::vec4{object.colors[3 * i], object.colors[3 * i + 1], object.colors[3 * i + 2], 1.f}
                            : color;
                        object.circles.push_back(create<aurora::AuroraCircle>(object.size, spot));
                        object.circles.back()->setLayer(BEHIND_EARTH_LAYER);  // under the coastlines
                    }
                    break;
            }
            if (!object.label.empty()) {
                object.text = create<aurora::AuroraText>(object.label, LABEL_SIZE, object.color.empty() ? theme.TEXT_PRIMARY : color);
                object.text->setLayer(POINT_LAYER);
            }
            placeObject(object);
            objects.emplace(id, std::move(object));
        }

        void destroyObject(Object& object) {
            for (const auto& segment : object.segments) destroy(segment);
            for (const auto& circle : object.circles) destroy(circle);
            if (object.text) destroy(object.text);
        }

        void clear() {
            for (auto& [id, object] : objects) destroyObject(object);
            objects.clear();
            for (const auto& coastline : coastlines) {
                for (const auto& segment : coastline.segments) destroy(segment);
            }
            coastlines.clear();
            for (const auto& line : globe_coastlines) {
                for (const auto& segment : line.segments) destroy(segment);
            }
            globe_coastlines.clear();
            has_earth = false;
            space_title->setText(" ");
            map_title->setText(" ");
            newPanel();
            window_size = {0.f, 0.f};
        }

        // The panel on the left, then the space view and the map side by side in the rest of the window
        void layout() {
            glm::vec2 size{getRenderer().getWidth(), getRenderer().getHeight()};
            if (size == window_size) return;
            window_size = size;

            float left = PANEL_WIDTH + 2.f * MARGIN;
            float width = size.x - left - MARGIN;
            float height = size.y - 2.f * MARGIN - TITLE_SIZE * 2.f;
            float top = MARGIN + TITLE_SIZE * 2.f;

            space_view.shown = layout_mode != "map";
            map_view.shown = layout_mode != "space";
            if (layout_mode == "space") {
                float side = std::min(width, height);
                space_view = {{left + (width - side) / 2.f, top}, {side, side}, true};
            } else if (layout_mode == "map") {
                float map_width = std::min(width, 2.f * height);
                map_view = {{left + (width - map_width) / 2.f, top + (height - map_width / 2.f) / 2.f}, {map_width, map_width / 2.f}, true};
            } else {
                // Two rows: the square space view on top, the 2:1 map below
                float row = (height - MARGIN - TITLE_SIZE * 2.f) / 2.f;
                float side = std::min(width, row);
                float map_width = std::min(width, 2.f * row);
                space_view = {{left + (width - side) / 2.f, top}, {side, side}, true};
                map_view = {{left + (width - map_width) / 2.f, top + row + MARGIN + TITLE_SIZE * 2.f}, {map_width, map_width / 2.f}, true};
            }

            space_title->setHidden(!space_view.shown);
            map_title->setHidden(!map_view.shown);
            space_title->setPosition(space_view.origin - glm::vec2{0.f, TITLE_SIZE * 2.f});
            map_title->setPosition(map_view.origin - glm::vec2{0.f, TITLE_SIZE * 2.f});

            placeMapGrid();
            for (const auto& coastline : coastlines) placeCoastline(coastline);
            placeSpace();
        }

        // Pixels per km, and the center of the space view
        float spaceScale() const { return space_view.size.x / 2.f / extent; }
        glm::vec2 spaceCenter() const { return space_view.origin + space_view.size / 2.f; }

        // Orthographic projection of a view space point (km, relative to the camera center)
        glm::vec2 spacePoint(glm::vec3 view) const {
            return spaceCenter() + spaceScale() * glm::vec2{view.x, -view.y};
        }

        glm::vec3 toView(glm::vec3 point) const { return camera * (point - camera_center); }

        // Behind the plane through the Earth's center facing the viewer, under the opaque disk
        bool behindEarth(glm::vec3 view) const {
            return has_earth && view.z < toView(glm::vec3{0.f}).z;
        }

        // Equirectangular map: longitude -180..180 left to right, latitude 90..-90 top to bottom
        glm::vec2 mapPoint(glm::vec2 track) const {
            return map_view.origin + map_view.size * (0.5f + track / glm::vec2{360.f, -180.f});
        }

        void placeMapGrid() {
            glm::vec2 start = map_view.origin;
            glm::vec2 end = map_view.origin + map_view.size;
            for (size_t i = 0; i < meridians.size(); ++i) {
                float x = start.x + map_view.size.x * static_cast<float>(i) / static_cast<float>(meridians.size() - 1);
                meridians[i]->setHidden(!map_view.shown);
                meridians[i]->setEndpoints({x, start.y}, {x, end.y}, GRID_THICKNESS);
            }
            for (size_t i = 0; i < parallels.size(); ++i) {
                float y = start.y + map_view.size.y * static_cast<float>(i) / static_cast<float>(parallels.size() - 1);
                parallels[i]->setHidden(!map_view.shown);
                parallels[i]->setEndpoints({start.x, y}, {end.x, y}, GRID_THICKNESS);
            }
        }

        // `values` alternates latitude and longitude
        void addCoastline(const std::vector<float>& values) {
            const auto& theme = aurora::AuroraThemeSettings::get();
            Coastline coastline;
            GlobeLine globe;
            for (size_t i = 0; i + 1 < values.size(); i += 2) {
                coastline.points.push_back({values[i + 1], values[i]});
                globe.units.push_back(unitVector(values[i], values[i + 1]));
            }
            for (size_t i = 1; i < coastline.points.size(); ++i) {
                auto segment = create<aurora::AuroraSegment>(theme.DELIMITER);
                segment->setLayer(SURFACE_LAYER);
                coastline.segments.push_back(segment);
            }
            placeCoastline(coastline);
            coastlines.push_back(std::move(coastline));
            addGlobeLine(globe_coastlines, std::move(globe), theme.DELIMITER);
            placeGlobeLine(globe_coastlines.back(), COASTLINE_THICKNESS);
        }

        void placeCoastline(const Coastline& coastline) {
            for (size_t i = 0; i < coastline.segments.size(); ++i) {
                coastline.segments[i]->setHidden(!map_view.shown);
                coastline.segments[i]->setEndpoints(mapPoint(coastline.points[i]), mapPoint(coastline.points[i + 1]), COASTLINE_THICKNESS);
            }
        }

        // Places the Earth and every object, for the current camera and Earth rotation
        void placeSpace() {
            bool globe = space_view.shown && has_earth;
            earth->setHidden(!globe);
            if (globe) {
                float radius = EARTH_RADIUS * spaceScale();
                earth->setPosition(spacePoint(toView(glm::vec3{0.f})));
                earth->setScale(radius, radius);
            }
            for (const auto& line : globe_grid) placeGlobeLine(line, GRID_THICKNESS);
            for (const auto& line : globe_coastlines) placeGlobeLine(line, COASTLINE_THICKNESS);
            for (auto& [id, object] : objects) placeObject(object);
        }

        // Hides the far side, and cuts the segments crossing the limb where they reach it
        void placeGlobeLine(const GlobeLine& line, float thickness) {
            bool shown = space_view.shown && has_earth && surface;
            glm::mat3 earth_to_view = camera * earth_to_space;
            glm::vec2 center = spacePoint(toView(glm::vec3{0.f}));
            float radius = EARTH_RADIUS * spaceScale();
            for (size_t i = 0; i < line.segments.size(); ++i) {
                glm::vec3 from = earth_to_view * line.units[i];
                glm::vec3 to = earth_to_view * line.units[i + 1];
                bool hidden = !shown || (from.z < 0.f && to.z < 0.f);
                line.segments[i]->setHidden(hidden);
                if (hidden) continue;
                if (from.z < 0.f) from = glm::normalize(glm::mix(from, to, from.z / (from.z - to.z)));
                else if (to.z < 0.f) to = glm::normalize(glm::mix(to, from, to.z / (to.z - from.z)));
                line.segments[i]->setEndpoints(center + radius * glm::vec2{from.x, -from.y}, center + radius * glm::vec2{to.x, -to.y}, thickness);
            }
        }

        void placeObject(Object& object) {
            glm::vec2 label_at{0.f};
            switch (object.kind) {
                case Kind::Line:
                    for (size_t i = 0; i < object.segments.size(); ++i) {
                        glm::vec3 from = toView(object.points[i]);
                        glm::vec3 to = toView(object.points[i + 1]);
                        object.segments[i]->setHidden(!space_view.shown);
                        object.segments[i]->setLayer(behindEarth((from + to) / 2.f) ? BEHIND_EARTH_LAYER : OBJECT_LAYER);
                        object.segments[i]->setEndpoints(spacePoint(from), spacePoint(to), object.size);
                    }
                    if (!object.points.empty()) label_at = spacePoint(toView(object.points.back()));
                    break;
                case Kind::Point: {
                    glm::vec3 view = toView(object.points[0]);
                    object.circles[0]->setHidden(!space_view.shown);
                    object.circles[0]->setLayer(behindEarth(view) ? BEHIND_EARTH_LAYER : POINT_LAYER);
                    label_at = spacePoint(view);
                    object.circles[0]->setPosition(label_at);
                    break;
                }
                case Kind::Dots:
                    for (size_t i = 0; i < object.circles.size(); ++i) {
                        glm::vec3 view = toView(object.points[i]);
                        object.circles[i]->setHidden(!space_view.shown);
                        object.circles[i]->setLayer(behindEarth(view) ? BEHIND_EARTH_LAYER : POINT_LAYER);
                        object.circles[i]->setPosition(spacePoint(view));
                    }
                    break;
                case Kind::Track:
                    for (size_t i = 0; i < object.segments.size(); ++i) {
                        glm::vec2 from = object.points[i];
                        glm::vec2 to = object.points[i + 1];
                        object.segments[i]->setHidden(!map_view.shown || std::abs(to.x - from.x) > JUMP);
                        object.segments[i]->setEndpoints(mapPoint(from), mapPoint(to), object.size);
                    }
                    if (!object.points.empty()) label_at = mapPoint(object.points.back());
                    break;
                case Kind::Spots:
                    for (size_t i = 0; i < object.circles.size(); ++i) {
                        object.circles[i]->setHidden(!map_view.shown);
                        object.circles[i]->setPosition(mapPoint(object.points[i]));
                    }
                    break;
            }
            if (object.text) {
                object.text->setHidden(object.kind == Kind::Track ? !map_view.shown : !space_view.shown);
                object.text->setPosition(label_at + glm::vec2{object.size + 6.f, -LABEL_SIZE});
            }
        }

        void sendCommand(const std::string& command) {
            server.broadcast(aurora::debug::CommandMessage{command}.toMessage());
        }

        std::shared_ptr<aurora::AuroraCircle> earth;
        std::shared_ptr<aurora::AuroraText> space_title;
        std::shared_ptr<aurora::AuroraText> map_title;
        std::shared_ptr<aurora::AuroraPanel> panel;

        std::vector<std::shared_ptr<aurora::AuroraSegment>> meridians;
        std::vector<std::shared_ptr<aurora::AuroraSegment>> parallels;
        std::vector<GlobeLine> globe_grid;
        std::vector<GlobeLine> globe_coastlines;
        std::vector<Coastline> coastlines;
        std::map<std::string, Object> objects;

        std::string layout_mode = "both";
        bool has_earth = false;
        bool surface = true;
        glm::mat3 earth_to_space{1.f};
        glm::mat3 camera{1.f};  // space to view: x right, y up, z toward the viewer
        glm::vec3 camera_center{0.f};
        float extent = 8000.f;  // km from the center to the edge of the space view

        aurora::debug::AuroraDebugServer server{PORT};

        View space_view{};
        View map_view{};
        glm::vec2 window_size{0.f, 0.f};
};

int main() {
    aurora::log::init(spdlog::level::info);

    OrbitViewer app{};

    try {
        app.run();
    } catch (const std::exception& e) {
        aurora::log::engine()->error("{}", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
